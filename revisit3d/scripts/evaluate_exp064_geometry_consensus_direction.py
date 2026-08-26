#!/usr/bin/env python3
"""Fixed decoded-geometry consensus direction audit for EXP-064."""
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

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import _model_view
from revisit3d.scripts.evaluate_exp062_order_sensitivity_anatomy import _camera_points
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _run_order(
    carrier: FrozenCUT3RCarrier,
    gt_views: list[dict],
    order: tuple[int, int, int],
) -> np.ndarray:
    state = None
    with torch.no_grad():
        for process_index, view_index in enumerate((0,) + order):
            _, state, _ = carrier.step(
                _model_view(gt_views[view_index], process_index), state
            )
        query = _model_view(gt_views[4], 4)
        query["update"] = torch.tensor([False])
        prediction, _, _ = carrier.step(query, state)
    return (
        prediction["pts3d_in_self_view"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _score(points: np.ndarray, target: np.ndarray, depth: np.ndarray, minimum: float) -> float:
    if not (np.isfinite(points).all() and np.all(points[:, 2] > 1e-6)):
        raise RuntimeError("EXP-064 consensus direction produced invalid points")
    scale = float(np.median(depth) / np.median(points[:, 2]))
    error = np.linalg.norm(scale * points - target, axis=-1)
    return float(np.mean(error / np.maximum(depth, minimum)))


def _bootstrap(
    rows: list[dict], key: str, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in rows})
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        means = []
        for scene in scenes:
            scene_rows = by_scene[scene]
            indices = rng.integers(0, len(scene_rows), size=len(scene_rows))
            means.append(np.mean([scene_rows[index][key] for index in indices]))
        draws[sample] = np.mean(means)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _scene_means(rows: list[dict], key: str) -> dict[str, float]:
    return {
        scene: float(np.mean([row[key] for row in rows if row["scene"] == scene]))
        for scene in sorted({row["scene"] for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-064_geometry_consensus_direction_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-direction", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_direction or not torch.cuda.is_available():
        raise SystemExit("EXP-064 requires train-RGB-D direction confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-064 result already exists")
    source_config_path = Path(config["source"]["exp062_config"])
    source_result_path = Path(config["source"]["exp062_result"])
    exp063_path = Path(config["source"]["exp063_result"])
    if not (
        _sha256(source_config_path) == config["source"]["exp062_config_sha256"]
        and _sha256(source_result_path) == config["source"]["exp062_result_sha256"]
        and _sha256(exp063_path) == config["source"]["exp063_result_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-064 immutable source contract failed")
    source_config = yaml.safe_load(source_config_path.read_text())
    source_result = json.loads(source_result_path.read_text())
    exp063 = json.loads(exp063_path.read_text())
    checkpoint = Path(source_config["carrier"]["checkpoint"])
    if not (
        source_result["passed"] is True
        and exp063["passed"] is False
        and exp063["gates"]["minimum_geometry_range_spearman"] is True
        and exp063["gates"]["minimum_geometry_over_latent_spearman"] is True
        and _sha256(checkpoint) == source_config["carrier"]["checkpoint_sha256"]
    ):
        raise RuntimeError("EXP-064 source qualification failed")

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
    chronological = (1, 2, 3)
    chronological_index = permutations.index(chronological)
    alpha = float(config["direction"]["interpolation"])
    rng = np.random.default_rng(int(config["direction"]["spatial_control_seed"]))
    minimum_depth = float(source_config["metric"]["minimum_depth_m"])
    maximum_depth = float(source_config["metric"]["maximum_depth_m"])
    minimum_valid = int(source_config["metric"]["minimum_valid_pixels"])
    source_orders = {
        (row["sequence"], int(row["query_frame"]), tuple(row["middle_order"])): row
        for row in source_result["order_rows"]
    }
    rows = []
    maximum_reproduction = 0.0
    torch.cuda.reset_peak_memory_stats()

    for sequence in source_config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for query_frame in source_config["data"]["query_frames"]:
            indices = list(range(int(query_frame) - 4, int(query_frame) + 1))
            dataset = SevenScenes(
                split="train",
                ROOT=source_config["data"]["root"],
                resolution=tuple(source_config["carrier"]["resolution"]),
                tuple_list=[
                    sequence + " " + " ".join(f"{index:06d}" for index in indices)
                ],
                seed=source_seed,
            )
            gt_views = dataset[0]
            predictions = [_run_order(carrier, gt_views, order) for order in permutations]
            depth_map = np.asarray(gt_views[4]["depthmap"], dtype=np.float64)
            target_map = _camera_points(
                depth_map, np.asarray(gt_views[4]["camera_intrinsics"], dtype=np.float64)
            )
            valid = (
                np.isfinite(target_map).all(axis=-1)
                & (depth_map >= minimum_depth)
                & (depth_map <= maximum_depth)
            )
            for prediction in predictions:
                valid &= np.isfinite(prediction).all(axis=-1) & (prediction[..., 2] > 1e-6)
            if int(valid.sum()) < minimum_valid:
                raise RuntimeError("EXP-064 has insufficient common query pixels")
            depth = depth_map[valid]
            target = target_map[valid]
            normalized = [
                prediction[valid] / float(np.median(prediction[..., 2][valid]))
                for prediction in predictions
            ]
            consensus = np.mean(np.stack(normalized), axis=0)
            base_errors = []
            consensus_errors = []
            for order, points in zip(permutations, normalized, strict=True):
                base = _score(points, target, depth, minimum_depth)
                moved = _score(points + alpha * (consensus - points), target, depth, minimum_depth)
                base_errors.append(base)
                consensus_errors.append(moved)
                expected = float(
                    source_orders[(sequence, int(query_frame), order)]["relative_3d_epe"]
                )
                maximum_reproduction = max(maximum_reproduction, abs(base - expected))

            chronological_points = normalized[chronological_index]
            residual = consensus - chronological_points
            shuffled = chronological_points + alpha * residual[
                rng.permutation(residual.shape[0])
            ]
            shuffled_error = _score(shuffled, target, depth, minimum_depth)
            chronological_base = base_errors[chronological_index]
            chronological_consensus = consensus_errors[chronological_index]
            row = {
                "scene": scene,
                "sequence": sequence,
                "query_frame": int(query_frame),
                "chronological_base_epe": chronological_base,
                "chronological_consensus_epe": chronological_consensus,
                "chronological_spatial_shuffle_epe": shuffled_error,
                "chronological_gain": chronological_base - chronological_consensus,
                "spatial_control_gain": shuffled_error - chronological_consensus,
                "mean_order_base_epe": float(np.mean(base_errors)),
                "mean_order_consensus_epe": float(np.mean(consensus_errors)),
                "mean_order_gain": float(np.mean(base_errors) - np.mean(consensus_errors)),
                "valid_pixels": int(valid.sum()),
                "interpolation": alpha,
            }
            rows.append(row)
            print(
                f"[{len(rows):02d}/16] {sequence}:{query_frame} "
                f"chrono_gain={row['chronological_gain']:+.6f} "
                f"shuffle_gain={row['spatial_control_gain']:+.6f} "
                f"all_gain={row['mean_order_gain']:+.6f}",
                flush=True,
            )
            del predictions, normalized
            gc.collect()
            torch.cuda.empty_cache()

    scene_chronological = _scene_means(rows, "chronological_gain")
    scene_spatial = _scene_means(rows, "spatial_control_gain")
    scene_all_order = _scene_means(rows, "mean_order_gain")
    samples = int(config["statistics"]["bootstrap_samples"])
    chronological_interval = _bootstrap(
        rows, "chronological_gain", samples, int(config["seed"]) + 1
    )
    spatial_interval = _bootstrap(
        rows, "spatial_control_gain", samples, int(config["seed"]) + 2
    )
    harm_fraction = float(np.mean([row["chronological_gain"] < 0 for row in rows]))
    summary = {
        "mean_chronological_base_epe": float(
            np.mean([row["chronological_base_epe"] for row in rows])
        ),
        "mean_chronological_consensus_epe": float(
            np.mean([row["chronological_consensus_epe"] for row in rows])
        ),
        "mean_chronological_spatial_shuffle_epe": float(
            np.mean([row["chronological_spatial_shuffle_epe"] for row in rows])
        ),
        "mean_chronological_gain": float(np.mean([row["chronological_gain"] for row in rows])),
        "mean_spatial_control_gain": float(np.mean([row["spatial_control_gain"] for row in rows])),
        "mean_all_order_gain": float(np.mean([row["mean_order_gain"] for row in rows])),
        "chronological_harm_fraction": harm_fraction,
        "maximum_base_reproduction_difference": maximum_reproduction,
    }
    tolerance = float(config["metric"]["maximum_reproduction_difference"])
    success = config["success"]
    gates = {
        "base_reproduced": maximum_reproduction <= tolerance,
        "chronological_gain_all_scenes": all(value > 0 for value in scene_chronological.values()),
        "chronological_bootstrap_lower_positive": chronological_interval[0] > 0,
        "spatial_control_gain_all_scenes": all(value > 0 for value in scene_spatial.values()),
        "spatial_control_bootstrap_lower_positive": spatial_interval[0] > 0,
        "maximum_chronological_harm_fraction": harm_fraction
        <= float(success["maximum_chronological_harm_fraction"]),
        "all_order_gain_all_scenes": all(value > 0 for value in scene_all_order.values()),
    }
    result = {
        "experiment": "EXP-064",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_exp063_sha256": _sha256(exp063_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_accessed": False,
        "terminal_accessed": False,
        "fit_performed": False,
        "summary": summary,
        "scene_chronological_gains": scene_chronological,
        "scene_spatial_control_gains": scene_spatial,
        "scene_all_order_gains": scene_all_order,
        "chronological_gain_bootstrap_95": chronological_interval,
        "spatial_control_gain_bootstrap_95": spatial_interval,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("summary", "scene_chronological_gains", "scene_spatial_control_gains", "scene_all_order_gains", "chronological_gain_bootstrap_95", "spatial_control_gain_bootstrap_95", "gates", "passed")}, indent=2))


if __name__ == "__main__":
    main()
