#!/usr/bin/env python3
"""Train-only calibration-shock state-poisoning anatomy for EXP-065."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import _model_view
from revisit3d.scripts.evaluate_exp062_order_sensitivity_anatomy import _camera_points
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _intervention_view(view: dict, index: int, condition: str, keep: float) -> dict:
    model_view = _model_view(view, index)
    image = model_view["img"]
    height, width = image.shape[-2:]
    crop_height = int(round(height * keep))
    crop_width = int(round(width * keep))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    bottom = top + crop_height
    right = left + crop_width

    if condition.startswith("zoom"):
        image = F.interpolate(
            image[..., top:bottom, left:right],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    elif condition == "resample_write":
        image = F.interpolate(
            F.interpolate(
                image,
                size=(crop_height, crop_width),
                mode="bilinear",
                align_corners=False,
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    elif condition == "periphery_mask_write":
        center = image[..., top:bottom, left:right].clone()
        image = image.mean(dim=(-2, -1), keepdim=True).expand_as(image).clone()
        image[..., top:bottom, left:right] = center
    elif condition not in {"clean_write", "clean_skip"}:
        raise ValueError(f"unknown intervention condition: {condition}")

    model_view["img"] = image.contiguous()
    model_view["update"] = torch.tensor([not condition.endswith("skip")])
    return model_view


def _query(
    carrier: FrozenCUT3RCarrier, view: dict, index: int, state
) -> np.ndarray:
    query = _model_view(view, index)
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


def _run_condition(
    carrier: FrozenCUT3RCarrier,
    views: list[dict],
    condition: str,
    keep: float,
) -> tuple[np.ndarray, np.ndarray]:
    state = None
    with torch.no_grad():
        for index in (0, 1):
            _, state, _ = carrier.step(_model_view(views[index], index), state)
        _, state, _ = carrier.step(
            _intervention_view(views[2], 2, condition, keep), state
        )

    immediate = _query(carrier, views[3], 3, state)
    with torch.no_grad():
        _, recovered_state, _ = carrier.step(_model_view(views[3], 3), state)
    persistent = _query(carrier, views[4], 4, recovered_state)
    return immediate, persistent


def _metric_errors(
    predictions: dict[str, np.ndarray],
    view: dict,
    minimum_depth: float,
    maximum_depth: float,
    minimum_valid: int,
) -> tuple[dict[str, float], int]:
    depth = np.asarray(view["depthmap"], dtype=np.float64)
    intrinsics = np.asarray(view["camera_intrinsics"], dtype=np.float64)
    target = _camera_points(depth, intrinsics)
    valid = (
        np.isfinite(target).all(axis=-1)
        & (depth >= minimum_depth)
        & (depth <= maximum_depth)
    )
    for prediction in predictions.values():
        valid &= np.isfinite(prediction).all(axis=-1) & (prediction[..., 2] > 1e-6)
    if int(valid.sum()) < minimum_valid:
        raise RuntimeError("EXP-065 has insufficient common metric pixels")

    errors: dict[str, float] = {}
    for condition, prediction in predictions.items():
        scale = float(np.median(depth[valid]) / np.median(prediction[..., 2][valid]))
        relative = np.linalg.norm(
            scale * prediction[valid] - target[valid], axis=-1
        ) / np.maximum(depth[valid], minimum_depth)
        errors[condition] = float(relative.mean())
    return errors, int(valid.sum())


def _stratified_interval(
    rows: list[dict], key: str, samples: int, seed: int
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
            scene_means.append(np.mean([scene_rows[index][key] for index in indices]))
        draws[sample] = np.mean(scene_means)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _scene_means(rows: list[dict], key: str) -> dict[str, float]:
    return {
        scene: float(np.mean([row[key] for row in rows if row["scene"] == scene]))
        for scene in sorted({row["scene"] for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-065_calibration_shock_anatomy_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-calibration-anatomy", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_calibration_anatomy or not torch.cuda.is_available():
        raise SystemExit("EXP-065 requires train-RGB-D anatomy confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-065 result already exists")
    manifest = json.loads(manifest_path.read_text())
    conditions = list(config["intervention"]["conditions"])
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and len(conditions) == int(config["success"]["exact_conditions"])
    ):
        raise RuntimeError("EXP-065 source-safe contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(config["carrier"]["repository"]).resolve()
    for path in (repository, repository / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
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

    keep = float(config["intervention"]["retained_axis_fraction"])
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    minimum_valid = int(config["metric"]["minimum_valid_pixels"])
    context_rows: list[dict] = []
    condition_rows: list[dict] = []
    replay_differences: list[float] = []
    skip_differences: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    total = len(config["data"]["sequences"]) * len(config["data"]["query_frames"])
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
            views = dataset[0]
            paths = {
                condition: _run_condition(carrier, views, condition, keep)
                for condition in conditions
            }
            clean_replay = _run_condition(carrier, views, "clean_write", keep)
            replay_difference = max(
                float(np.max(np.abs(paths["clean_write"][stage] - clean_replay[stage])))
                for stage in (0, 1)
            )
            skip_difference = max(
                float(np.max(np.abs(paths["clean_skip"][stage] - paths["zoom_skip"][stage])))
                for stage in (0, 1)
            )
            replay_differences.append(replay_difference)
            skip_differences.append(skip_difference)

            immediate_predictions = {key: value[0] for key, value in paths.items()}
            persistent_predictions = {key: value[1] for key, value in paths.items()}
            immediate, immediate_valid = _metric_errors(
                immediate_predictions, views[3], minimum_depth, maximum_depth, minimum_valid
            )
            persistent, persistent_valid = _metric_errors(
                persistent_predictions, views[4], minimum_depth, maximum_depth, minimum_valid
            )

            immediate_excess = (
                immediate["zoom_write"] - immediate["zoom_skip"]
            ) - (immediate["clean_write"] - immediate["clean_skip"])
            persistent_excess = (
                persistent["zoom_write"] - persistent["zoom_skip"]
            ) - (persistent["clean_write"] - persistent["clean_skip"])
            row = {
                "scene": scene,
                "sequence": sequence,
                "query_frame": int(query_frame),
                "frame_indices": frame_indices,
                "immediate_errors": immediate,
                "persistent_errors": persistent,
                "immediate_excess_write_penalty": float(immediate_excess),
                "persistent_excess_write_penalty": float(persistent_excess),
                "persistent_zoom_over_resample": float(
                    persistent["zoom_write"] - persistent["resample_write"]
                ),
                "persistent_zoom_over_mask": float(
                    persistent["zoom_write"] - persistent["periphery_mask_write"]
                ),
                "clean_replay_maximum_abs_difference": replay_difference,
                "skip_path_maximum_abs_difference": skip_difference,
                "immediate_valid_pixels": immediate_valid,
                "persistent_valid_pixels": persistent_valid,
            }
            context_rows.append(row)
            for condition in conditions:
                condition_rows.append(
                    {
                        "scene": scene,
                        "sequence": sequence,
                        "query_frame": int(query_frame),
                        "condition": condition,
                        "immediate_relative_3d_epe": immediate[condition],
                        "persistent_relative_3d_epe": persistent[condition],
                    }
                )
            print(
                f"[{len(context_rows):02d}/{total}] {sequence}:{query_frame} "
                f"immediate={immediate_excess:+.6f} persistent={persistent_excess:+.6f} "
                f"zoom-resample={row['persistent_zoom_over_resample']:+.6f}",
                flush=True,
            )
            del paths, clean_replay, immediate_predictions, persistent_predictions
            gc.collect()
            torch.cuda.empty_cache()

    keys = (
        "persistent_excess_write_penalty",
        "persistent_zoom_over_resample",
        "persistent_zoom_over_mask",
        "immediate_excess_write_penalty",
    )
    scene_summaries = {key: _scene_means(context_rows, key) for key in keys}
    intervals = {
        key: _stratified_interval(
            context_rows,
            key,
            int(config["statistics"]["bootstrap_samples"]),
            seed + offset + 1,
        )
        for offset, key in enumerate(keys)
    }
    mean_clean = float(
        np.mean([row["persistent_errors"]["clean_write"] for row in context_rows])
    )
    mean_persistent_excess = float(
        np.mean([row["persistent_excess_write_penalty"] for row in context_rows])
    )
    excess_fraction = mean_persistent_excess / max(mean_clean, 1e-12)
    positive_fraction = float(
        np.mean([row["persistent_excess_write_penalty"] > 0 for row in context_rows])
    )
    maximum_replay = max(replay_differences)
    maximum_skip = max(skip_differences)
    success = config["success"]
    gates = {
        "exact_counts": (
            len({row["scene"] for row in context_rows}) == int(success["exact_scenes"])
            and len(context_rows) == int(success["exact_contexts"])
            and len(condition_rows) == int(success["exact_contexts"]) * int(success["exact_conditions"])
        ),
        "clean_replay_within_tolerance": maximum_replay
        <= float(success["maximum_replay_abs_difference"]),
        "skip_paths_within_tolerance": maximum_skip
        <= float(success["maximum_skip_path_abs_difference"]),
        "persistent_excess_positive_all_scenes": all(
            value > 0 for value in scene_summaries["persistent_excess_write_penalty"].values()
        ),
        "persistent_excess_bootstrap_lower_positive": intervals[
            "persistent_excess_write_penalty"
        ][0]
        > 0,
        "minimum_persistent_excess_fraction": excess_fraction
        >= float(success["minimum_persistent_excess_fraction"]),
        "minimum_positive_context_fraction": positive_fraction
        >= float(success["minimum_positive_context_fraction"]),
        "zoom_over_resample_positive_all_scenes": all(
            value > 0 for value in scene_summaries["persistent_zoom_over_resample"].values()
        ),
        "zoom_over_resample_bootstrap_lower_positive": intervals[
            "persistent_zoom_over_resample"
        ][0]
        > 0,
        "zoom_over_mask_positive_all_scenes": all(
            value > 0 for value in scene_summaries["persistent_zoom_over_mask"].values()
        ),
        "zoom_over_mask_bootstrap_lower_positive": intervals[
            "persistent_zoom_over_mask"
        ][0]
        > 0,
        "immediate_and_persistent_positive_all_scenes": all(
            scene_summaries["immediate_excess_write_penalty"][scene] > 0
            and scene_summaries["persistent_excess_write_penalty"][scene] > 0
            for scene in scene_summaries["persistent_excess_write_penalty"]
        ),
    }
    result = {
        "experiment": "EXP-065",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(manifest_path),
        "fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "summary": {
            "mean_persistent_clean_write_epe": mean_clean,
            "mean_persistent_excess_write_penalty": mean_persistent_excess,
            "persistent_excess_fraction_of_clean_epe": excess_fraction,
            "positive_context_fraction": positive_fraction,
            "mean_immediate_excess_write_penalty": float(
                np.mean([row["immediate_excess_write_penalty"] for row in context_rows])
            ),
            "mean_persistent_zoom_over_resample": float(
                np.mean([row["persistent_zoom_over_resample"] for row in context_rows])
            ),
            "mean_persistent_zoom_over_mask": float(
                np.mean([row["persistent_zoom_over_mask"] for row in context_rows])
            ),
            "maximum_clean_replay_difference": maximum_replay,
            "maximum_skip_path_difference": maximum_skip,
        },
        "scene_summaries": scene_summaries,
        "bootstrap_95": intervals,
        "gates": gates,
        "passed": all(gates.values()),
        "rows": context_rows,
        "condition_rows": condition_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": result["summary"], "gates": gates, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()

