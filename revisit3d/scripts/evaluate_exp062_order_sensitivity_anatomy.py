#!/usr/bin/env python3
"""Fixed-evidence recurrent order-sensitivity anatomy for EXP-062."""
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
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return np.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ),
        axis=-1,
    )


def _run_order(
    carrier: FrozenCUT3RCarrier, gt_views: list[dict], middle_order: tuple[int, int, int]
) -> np.ndarray:
    state = None
    history_order = (0,) + middle_order
    with torch.no_grad():
        for process_index, view_index in enumerate(history_order):
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


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    x = _rank(np.asarray(first, dtype=np.float64))
    y = _rank(np.asarray(second, dtype=np.float64))
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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
            scene_means.append(np.mean([scene_rows[index]["absolute_range"] for index in indices]))
        draws[sample] = np.mean(scene_means)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-062_order_sensitivity_anatomy_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-order-anatomy", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_order_anatomy or not torch.cuda.is_available():
        raise SystemExit("EXP-062 requires train-RGB-D order-anatomy confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-062 result already exists")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    expected_depth_frames = (
        len(config["data"]["sequences"])
        * len(config["data"]["query_frames"])
        * (int(config["data"]["history_frames"]) + 1)
    )
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and preparation["selected_frames"] == expected_depth_frames
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and int(config["data"]["history_frames"]) == 4
    ):
        raise RuntimeError("EXP-062 source-safe contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(config["carrier"]["repository"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    if str(repository / "src") not in sys.path:
        sys.path.insert(0, str(repository / "src"))
    from eval.mv_recon.data import SevenScenes

    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=8,
        basis_seed=seed,
        update_type="ttt3r",
    ).cuda()
    carrier.eval()
    carrier.requires_grad_(False)

    permutations = list(itertools.permutations((1, 2, 3)))
    chronological = (1, 2, 3)
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    minimum_valid = int(config["metric"]["minimum_valid_pixels"])
    context_rows: list[dict] = []
    order_rows: list[dict] = []
    replay_rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for query_frame in config["data"]["query_frames"]:
            frame_indices = list(range(int(query_frame) - 4, int(query_frame) + 1))
            tuple_spec = sequence + " " + " ".join(
                f"{index:06d}" for index in frame_indices
            )
            dataset = SevenScenes(
                split="train",
                ROOT=config["data"]["root"],
                resolution=tuple(config["carrier"]["resolution"]),
                tuple_list=[tuple_spec],
                seed=seed,
            )
            gt_views = dataset[0]
            predictions = {
                order: _run_order(carrier, gt_views, order) for order in permutations
            }
            replay = _run_order(carrier, gt_views, chronological)
            replay_difference = float(
                np.max(np.abs(predictions[chronological] - replay))
            )
            replay_rows.append(
                {
                    "scene": scene,
                    "sequence": sequence,
                    "query_frame": int(query_frame),
                    "maximum_abs_point_difference": replay_difference,
                }
            )

            depth = np.asarray(gt_views[4]["depthmap"], dtype=np.float64)
            target = _camera_points(
                depth, np.asarray(gt_views[4]["camera_intrinsics"], dtype=np.float64)
            )
            common_valid = (
                np.isfinite(target).all(axis=-1)
                & (depth >= minimum_depth)
                & (depth <= maximum_depth)
            )
            for prediction in predictions.values():
                common_valid &= np.isfinite(prediction).all(axis=-1) & (
                    prediction[..., 2] > 1e-6
                )
            if int(common_valid.sum()) < minimum_valid:
                raise RuntimeError("EXP-062 has insufficient common query pixels")

            normalized_predictions = {}
            errors = {}
            for order, prediction in predictions.items():
                scale = float(
                    np.median(depth[common_valid])
                    / np.median(prediction[..., 2][common_valid])
                )
                relative = np.linalg.norm(
                    scale * prediction[common_valid] - target[common_valid], axis=-1
                ) / np.maximum(depth[common_valid], minimum_depth)
                error = float(np.mean(relative))
                errors[order] = error
                native_scale = float(np.median(prediction[..., 2][common_valid]))
                normalized_predictions[order] = prediction[common_valid] / native_scale
                order_rows.append(
                    {
                        "scene": scene,
                        "sequence": sequence,
                        "query_frame": int(query_frame),
                        "middle_order": list(order),
                        "frame_order": [
                            frame_indices[0],
                            *[frame_indices[index] for index in order],
                            frame_indices[4],
                        ],
                        "query_update": False,
                        "relative_3d_epe": error,
                        "scale": scale,
                        "valid_pixels": int(common_valid.sum()),
                    }
                )

            pairwise = []
            for first, second in itertools.combinations(permutations, 2):
                pairwise.append(
                    float(
                        np.mean(
                            np.linalg.norm(
                                normalized_predictions[first] - normalized_predictions[second],
                                axis=-1,
                            )
                        )
                    )
                )
            best_order = min(permutations, key=errors.__getitem__)
            worst_order = max(permutations, key=errors.__getitem__)
            absolute_range = errors[worst_order] - errors[best_order]
            relative_range = absolute_range / max(errors[chronological], 1e-12)
            context_row = {
                "scene": scene,
                "sequence": sequence,
                "query_frame": int(query_frame),
                "chronological_epe": errors[chronological],
                "best_epe": errors[best_order],
                "worst_epe": errors[worst_order],
                "best_middle_order": list(best_order),
                "worst_middle_order": list(worst_order),
                "absolute_range": absolute_range,
                "relative_range": relative_range,
                "prediction_dispersion": float(np.mean(pairwise)),
                "replay_maximum_abs_difference": replay_difference,
                "valid_pixels": int(common_valid.sum()),
            }
            context_rows.append(context_row)
            print(
                f"[{len(context_rows):02d}/16] {sequence}:{query_frame} "
                f"chrono={errors[chronological]:.5f} range={relative_range:.2%} "
                f"disp={context_row['prediction_dispersion']:.5f}",
                flush=True,
            )
            del predictions, replay, normalized_predictions
            gc.collect()
            torch.cuda.empty_cache()

    scene_ranges = {
        scene: float(
            np.mean([row["absolute_range"] for row in context_rows if row["scene"] == scene])
        )
        for scene in sorted({row["scene"] for row in context_rows})
    }
    mean_chronological = float(np.mean([row["chronological_epe"] for row in context_rows]))
    mean_absolute_range = float(np.mean([row["absolute_range"] for row in context_rows]))
    aggregate_relative_range = mean_absolute_range / mean_chronological
    threshold = float(config["success"]["minimum_context_relative_range"])
    context_fraction = float(
        np.mean([row["relative_range"] >= threshold for row in context_rows])
    )
    dispersion_spearman = _spearman(
        [row["prediction_dispersion"] for row in context_rows],
        [row["absolute_range"] for row in context_rows],
    )
    interval = _stratified_bootstrap(
        context_rows,
        int(config["statistics"]["bootstrap_samples"]),
        seed + 1,
    )
    maximum_replay = max(row["maximum_abs_point_difference"] for row in replay_rows)
    success = config["success"]
    gates = {
        "exact_counts": (
            len(scene_ranges) == int(success["exact_scenes"])
            and len(context_rows) == int(success["exact_contexts"])
            and len(order_rows) == int(success["exact_order_evaluations"])
            and len(replay_rows) == int(success["exact_replay_checks"])
        ),
        "replay_within_tolerance": maximum_replay
        <= float(success["maximum_replay_abs_difference"]),
        "range_positive_all_scenes": all(value > 0 for value in scene_ranges.values()),
        "bootstrap_lower_positive": interval[0] > 0,
        "minimum_aggregate_relative_range": aggregate_relative_range
        >= float(success["minimum_aggregate_relative_range"]),
        "minimum_context_fraction": context_fraction
        >= float(success["minimum_context_fraction"]),
        "minimum_dispersion_range_spearman": dispersion_spearman
        >= float(success["minimum_dispersion_range_spearman"]),
    }
    result = {
        "experiment": "EXP-062",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "depth_preparation_sha256": _sha256(preparation_path),
        "validation_accessed": False,
        "terminal_accessed": False,
        "first_history_fixed": True,
        "history_payload_identical": True,
        "query_identical_and_update_false": True,
        "summary": {
            "mean_chronological_epe": mean_chronological,
            "mean_best_epe": float(np.mean([row["best_epe"] for row in context_rows])),
            "mean_worst_epe": float(np.mean([row["worst_epe"] for row in context_rows])),
            "mean_absolute_range": mean_absolute_range,
            "aggregate_relative_range": aggregate_relative_range,
            "context_fraction_above_relative_threshold": context_fraction,
            "dispersion_absolute_range_spearman": dispersion_spearman,
            "maximum_replay_abs_difference": maximum_replay,
        },
        "scene_absolute_ranges": scene_ranges,
        "absolute_range_bootstrap_95": interval,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "context_rows": context_rows,
        "order_rows": order_rows,
        "replay_rows": replay_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("summary", "scene_absolute_ranges", "absolute_range_bootstrap_95", "gates", "passed")}, indent=2))


if __name__ == "__main__":
    main()
