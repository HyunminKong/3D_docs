#!/usr/bin/env python3
"""No-fit geometry-decoded commutator capacity audit for EXP-063."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenCUT3RCarrier, RecurrentCarrierState
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import _model_view
from revisit3d.scripts.evaluate_exp062_order_sensitivity_anatomy import (
    _camera_points,
    _spearman,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _history_state(
    carrier: FrozenCUT3RCarrier,
    gt_views: list[dict],
    middle_order: tuple[int, int, int],
) -> RecurrentCarrierState:
    state = None
    with torch.no_grad():
        for process_index, view_index in enumerate((0,) + middle_order):
            _, state, _ = carrier.step(
                _model_view(gt_views[view_index], process_index), state
            )
    assert state is not None
    return state


def _query(
    carrier: FrozenCUT3RCarrier,
    gt_view: dict,
    state: RecurrentCarrierState,
) -> np.ndarray:
    query = _model_view(gt_view, 4)
    query["update"] = torch.tensor([False])
    with torch.no_grad():
        prediction, _, _ = carrier.step(query, state)
    return (
        prediction["pts3d_in_self_view"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _barycenter(states: list[RecurrentCarrierState], chronological: int) -> RecurrentCarrierState:
    reference = states[chronological]
    for state in states:
        if not (
            torch.equal(state.state_pos, reference.state_pos)
            and torch.equal(state.init_state_feat, reference.init_state_feat)
            and torch.equal(state.init_mem, reference.init_mem)
            and state.previous_reset == reference.previous_reset
        ):
            raise RuntimeError("EXP-063 immutable recurrent-state fields diverged")
    return RecurrentCarrierState(
        state_feat=torch.stack([state.state_feat for state in states]).mean(dim=0),
        state_pos=reference.state_pos,
        init_state_feat=reference.init_state_feat,
        mem=torch.stack([state.mem for state in states]).mean(dim=0),
        init_mem=reference.init_mem,
        previous_reset=reference.previous_reset,
    )


def _relative_error(
    prediction: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    minimum_depth: float,
) -> tuple[float, float]:
    scale = float(
        np.median(depth[valid]) / np.median(prediction[..., 2][valid])
    )
    error = np.linalg.norm(scale * prediction[valid] - target[valid], axis=-1)
    return float(np.mean(error / np.maximum(depth[valid], minimum_depth))), scale


def _pairwise_dispersion(arrays: list[np.ndarray]) -> float:
    distances = []
    for first, second in itertools.combinations(arrays, 2):
        distances.append(
            float(np.mean(np.linalg.norm(first - second, axis=-1)))
        )
    return float(np.mean(distances))


def _normalized_tensor_dispersion(
    tensors: list[torch.Tensor], chronological: int
) -> float:
    magnitude = float(tensors[chronological].float().square().mean().sqrt().item())
    if magnitude <= 1e-12:
        raise RuntimeError("EXP-063 chronological latent magnitude is degenerate")
    values = []
    for first, second in itertools.combinations(tensors, 2):
        values.append(
            float((first.float() - second.float()).square().mean().sqrt().item())
            / magnitude
        )
    return float(np.mean(values))


def _stratified_bootstrap(
    rows: list[dict], samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in rows})
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        scene_means = []
        for scene in scenes:
            scene_rows = by_scene[scene]
            indices = rng.integers(0, len(scene_rows), size=len(scene_rows))
            scene_means.append(
                np.mean([scene_rows[index]["barycenter_gain_over_order_mean"] for index in indices])
            )
        draws[sample] = np.mean(scene_means)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-063_geometry_commutator_capacity_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-capacity", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_capacity or not torch.cuda.is_available():
        raise SystemExit("EXP-063 requires train-RGB-D capacity confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-063 result already exists")
    source_config_path = Path(config["source"]["exp062_config"])
    source_result_path = Path(config["source"]["exp062_result"])
    preparation_path = Path(config["source"]["depth_preparation"])
    if not (
        _sha256(source_config_path) == config["source"]["exp062_config_sha256"]
        and _sha256(source_result_path) == config["source"]["exp062_result_sha256"]
        and _sha256(preparation_path) == config["source"]["depth_preparation_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-063 immutable source contract failed")
    source_config = yaml.safe_load(source_config_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    manifest_path = Path(source_config["data"]["train_manifest"])
    checkpoint = Path(source_config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == source_config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == source_config["carrier"]["checkpoint_sha256"]
        and source_result["passed"] is True
        and source_result["validation_accessed"] is False
        and source_result["terminal_accessed"] is False
    ):
        raise RuntimeError("EXP-063 EXP-062 qualification contract failed")

    source_seed = int(source_config["seed"])
    torch.manual_seed(source_seed)
    torch.cuda.manual_seed_all(source_seed)
    np.random.seed(source_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(source_config["carrier"]["repository"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    if str(repository / "src") not in sys.path:
        sys.path.insert(0, str(repository / "src"))
    from eval.mv_recon.data import SevenScenes

    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=8,
        basis_seed=source_seed,
        update_type="ttt3r",
    ).cuda()
    carrier.eval()
    carrier.requires_grad_(False)

    permutations = list(itertools.permutations((1, 2, 3)))
    chronological_order = (1, 2, 3)
    chronological_index = permutations.index(chronological_order)
    minimum_depth = float(source_config["metric"]["minimum_depth_m"])
    maximum_depth = float(source_config["metric"]["maximum_depth_m"])
    minimum_valid = int(source_config["metric"]["minimum_valid_pixels"])
    source_contexts = {
        (row["sequence"], int(row["query_frame"])): row
        for row in source_result["context_rows"]
    }
    source_orders = {
        (row["sequence"], int(row["query_frame"]), tuple(row["middle_order"])): row
        for row in source_result["order_rows"]
    }
    rows: list[dict] = []
    maximum_epe_reproduction = 0.0
    maximum_dispersion_reproduction = 0.0
    torch.cuda.reset_peak_memory_stats()

    for sequence in source_config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for query_frame in source_config["data"]["query_frames"]:
            indices = list(range(int(query_frame) - 4, int(query_frame) + 1))
            tuple_spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
            dataset = SevenScenes(
                split="train",
                ROOT=source_config["data"]["root"],
                resolution=tuple(source_config["carrier"]["resolution"]),
                tuple_list=[tuple_spec],
                seed=source_seed,
            )
            gt_views = dataset[0]
            states = [_history_state(carrier, gt_views, order) for order in permutations]
            predictions = [_query(carrier, gt_views[4], state) for state in states]
            barycenter_state = _barycenter(states, chronological_index)
            barycenter_prediction = _query(carrier, gt_views[4], barycenter_state)
            output_barycenter_prediction = np.mean(np.stack(predictions), axis=0)

            depth = np.asarray(gt_views[4]["depthmap"], dtype=np.float64)
            target = _camera_points(
                depth, np.asarray(gt_views[4]["camera_intrinsics"], dtype=np.float64)
            )
            valid = (
                np.isfinite(target).all(axis=-1)
                & (depth >= minimum_depth)
                & (depth <= maximum_depth)
            )
            for prediction in predictions:
                valid &= np.isfinite(prediction).all(axis=-1) & (prediction[..., 2] > 1e-6)
            if int(valid.sum()) < minimum_valid:
                raise RuntimeError("EXP-063 has insufficient common query pixels")
            if not (
                np.isfinite(barycenter_prediction[valid]).all()
                and np.isfinite(output_barycenter_prediction[valid]).all()
                and np.all(barycenter_prediction[..., 2][valid] > 1e-6)
                and np.all(output_barycenter_prediction[..., 2][valid] > 1e-6)
            ):
                raise RuntimeError("EXP-063 barycenter prediction is invalid")

            errors = []
            normalized = []
            for order, prediction in zip(permutations, predictions, strict=True):
                error, _ = _relative_error(prediction, target, depth, valid, minimum_depth)
                errors.append(error)
                native_scale = float(np.median(prediction[..., 2][valid]))
                normalized.append(prediction[valid] / native_scale)
                expected = float(
                    source_orders[(sequence, int(query_frame), order)]["relative_3d_epe"]
                )
                maximum_epe_reproduction = max(
                    maximum_epe_reproduction, abs(error - expected)
                )
            geometry_dispersion = _pairwise_dispersion(normalized)
            source_context = source_contexts[(sequence, int(query_frame))]
            maximum_dispersion_reproduction = max(
                maximum_dispersion_reproduction,
                abs(geometry_dispersion - float(source_context["prediction_dispersion"])),
            )
            barycenter_epe, _ = _relative_error(
                barycenter_prediction, target, depth, valid, minimum_depth
            )
            output_barycenter_epe, _ = _relative_error(
                output_barycenter_prediction, target, depth, valid, minimum_depth
            )
            order_mean = float(np.mean(errors))
            latent_dispersion = _normalized_tensor_dispersion(
                [state.state_feat for state in states], chronological_index
            )
            memory_dispersion = _normalized_tensor_dispersion(
                [state.mem for state in states], chronological_index
            )
            row = {
                "scene": scene,
                "sequence": sequence,
                "query_frame": int(query_frame),
                "chronological_epe": errors[chronological_index],
                "mean_order_epe": order_mean,
                "absolute_metric_range": max(errors) - min(errors),
                "geometry_dispersion": geometry_dispersion,
                "latent_state_dispersion": latent_dispersion,
                "pose_memory_dispersion": memory_dispersion,
                "state_barycenter_epe": barycenter_epe,
                "output_barycenter_epe": output_barycenter_epe,
                "barycenter_gain_over_order_mean": order_mean - barycenter_epe,
                "barycenter_gain_over_chronological": errors[chronological_index]
                - barycenter_epe,
                "output_barycenter_gain_over_order_mean": order_mean
                - output_barycenter_epe,
                "valid_pixels": int(valid.sum()),
            }
            rows.append(row)
            print(
                f"[{len(rows):02d}/16] {sequence}:{query_frame} "
                f"mean={order_mean:.5f} state_bar={barycenter_epe:.5f} "
                f"geo={geometry_dispersion:.5f} latent={latent_dispersion:.5f}",
                flush=True,
            )
            del states, predictions, barycenter_state
            gc.collect()
            torch.cuda.empty_cache()

    geometry_spearman = _spearman(
        [row["geometry_dispersion"] for row in rows],
        [row["absolute_metric_range"] for row in rows],
    )
    latent_spearman = _spearman(
        [row["latent_state_dispersion"] for row in rows],
        [row["absolute_metric_range"] for row in rows],
    )
    memory_spearman = _spearman(
        [row["pose_memory_dispersion"] for row in rows],
        [row["absolute_metric_range"] for row in rows],
    )
    scene_gains = {
        scene: float(
            np.mean(
                [
                    row["barycenter_gain_over_order_mean"]
                    for row in rows
                    if row["scene"] == scene
                ]
            )
        )
        for scene in sorted({row["scene"] for row in rows})
    }
    interval = _stratified_bootstrap(
        rows,
        int(config["statistics"]["bootstrap_samples"]),
        int(config["seed"]) + 1,
    )
    summary = {
        "mean_chronological_epe": float(np.mean([row["chronological_epe"] for row in rows])),
        "mean_order_epe": float(np.mean([row["mean_order_epe"] for row in rows])),
        "mean_state_barycenter_epe": float(np.mean([row["state_barycenter_epe"] for row in rows])),
        "mean_output_barycenter_epe": float(np.mean([row["output_barycenter_epe"] for row in rows])),
        "mean_barycenter_gain_over_order_mean": float(
            np.mean([row["barycenter_gain_over_order_mean"] for row in rows])
        ),
        "mean_barycenter_gain_over_chronological": float(
            np.mean([row["barycenter_gain_over_chronological"] for row in rows])
        ),
        "geometry_range_spearman": geometry_spearman,
        "latent_range_spearman": latent_spearman,
        "pose_memory_range_spearman": memory_spearman,
        "geometry_over_latent_spearman": geometry_spearman - latent_spearman,
        "maximum_epe_reproduction_difference": maximum_epe_reproduction,
        "maximum_geometry_dispersion_reproduction_difference": maximum_dispersion_reproduction,
    }
    tolerance = float(config["metric"]["maximum_reproduction_difference"])
    success = config["success"]
    gates = {
        "exp062_reproduced": max(maximum_epe_reproduction, maximum_dispersion_reproduction)
        <= tolerance,
        "minimum_geometry_range_spearman": geometry_spearman
        >= float(success["minimum_geometry_range_spearman"]),
        "minimum_geometry_over_latent_spearman": geometry_spearman - latent_spearman
        >= float(success["minimum_geometry_over_latent_spearman"]),
        "barycenter_gain_all_scenes": all(value > 0 for value in scene_gains.values()),
        "barycenter_bootstrap_lower_positive": interval[0] > 0,
        "barycenter_not_worse_than_chronological": summary[
            "mean_state_barycenter_epe"
        ]
        <= summary["mean_chronological_epe"],
    }
    result = {
        "experiment": "EXP-063",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_result_sha256": _sha256(source_result_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_accessed": False,
        "terminal_accessed": False,
        "fit_performed": False,
        "summary": summary,
        "scene_barycenter_gains": scene_gains,
        "barycenter_gain_bootstrap_95": interval,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("summary", "scene_barycenter_gains", "barycenter_gain_bootstrap_95", "gates", "passed")}, indent=2))


if __name__ == "__main__":
    main()
