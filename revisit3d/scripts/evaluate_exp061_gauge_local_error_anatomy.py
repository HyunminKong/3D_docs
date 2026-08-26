#!/usr/bin/env python3
"""Zero-fit held-pixel gauge/local error anatomy for EXP-061."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import _model_view
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


def _umeyama(source: np.ndarray, target: np.ndarray) -> Sim3:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Sim(3) inputs must have matched [N,3] shape")
    if source.shape[0] < 4:
        raise ValueError("Sim(3) requires at least four correspondences")
    source = source.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_t
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if not np.isfinite(variance) or variance <= 1e-12:
        raise RuntimeError("degenerate Sim(3) source variance")
    scale = float(np.sum(singular * signs) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if not (
        np.isfinite(scale)
        and scale > 0
        and np.isfinite(rotation).all()
        and np.isfinite(translation).all()
    ):
        raise RuntimeError("non-finite Sim(3) solution")
    return Sim3(scale, rotation.astype(np.float64), translation.astype(np.float64))


def _apply(transform: Sim3, points: np.ndarray) -> np.ndarray:
    return transform.scale * (points @ transform.rotation.T) + transform.translation


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


def _world_points(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    camera = _camera_points(depth.astype(np.float64), intrinsics.astype(np.float64))
    return camera @ pose[:3, :3].T + pose[:3, 3]


def _subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if points.shape[0] <= maximum:
        return points
    indices = np.linspace(0, points.shape[0] - 1, maximum, dtype=np.int64)
    return points[indices]


def _relative_epe(
    transform: Sim3,
    predicted: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> float:
    transformed = _apply(transform, predicted[mask])
    error = np.linalg.norm(transformed - target[mask], axis=-1)
    return float(np.mean(error / np.maximum(depth[mask], 1e-6)))


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


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


def _scene_means(rows: list[dict], key: str) -> dict[str, float]:
    return {
        scene: float(np.mean([row[key] for row in rows if row["scene"] == scene]))
        for scene in sorted({row["scene"] for row in rows})
    }


def _stratified_bootstrap(
    rows: list[dict], first_key: str, second_key: str, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in rows})
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        scene_differences = []
        for scene in scenes:
            scene_rows = by_scene[scene]
            indices = rng.integers(0, len(scene_rows), size=len(scene_rows))
            scene_differences.append(
                np.mean(
                    [
                        scene_rows[index][first_key] - scene_rows[index][second_key]
                        for index in indices
                    ]
                )
            )
        draws[sample] = np.mean(scene_differences)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-061_gauge_local_error_anatomy_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-anatomy", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_anatomy or not torch.cuda.is_available():
        raise SystemExit("EXP-061 requires train-RGB-D anatomy confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-061 result already exists")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    expected_frames = (
        len(config["data"]["sequences"])
        * len(config["data"]["target_frames"])
        * int(config["data"]["context_frames"])
    )
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and preparation["selected_frames"] == expected_frames
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-061 source-safe contract failed")

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

    metric = config["metric"]
    minimum_depth = float(metric["minimum_depth_m"])
    maximum_depth = float(metric["maximum_depth_m"])
    fit_parity = int(metric["fit_checkerboard_parity"])
    evaluation_parity = int(metric["evaluation_checkerboard_parity"])
    maximum_fit = int(metric["maximum_fit_points_per_frame"])
    minimum_fit = int(metric["minimum_fit_points_per_frame"])
    minimum_evaluation = int(metric["minimum_evaluation_points_per_frame"])
    high_quantile = float(metric["high_confidence_quantile"])
    context_frames = int(config["data"]["context_frames"])
    context_rows: list[dict] = []
    frame_rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame in config["data"]["target_frames"]:
            indices = list(
                range(int(target_frame) - context_frames + 1, int(target_frame) + 1)
            )
            tuple_spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
            dataset = SevenScenes(
                split="train",
                ROOT=config["data"]["root"],
                resolution=tuple(config["carrier"]["resolution"]),
                tuple_list=[tuple_spec],
                seed=seed,
            )
            gt_views = dataset[0]
            state = None
            frames = []
            with torch.no_grad():
                for local_index, gt_view in enumerate(gt_views):
                    prediction, state, _ = carrier.step(
                        _model_view(gt_view, local_index), state
                    )
                    predicted = (
                        prediction["pts3d_in_other_view"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    confidence = (
                        prediction["conf"][0].detach().float().cpu().numpy().astype(np.float64)
                    )
                    depth = np.asarray(gt_view["depthmap"], dtype=np.float64)
                    target = _world_points(
                        depth,
                        np.asarray(gt_view["camera_intrinsics"]),
                        np.asarray(gt_view["camera_pose"]),
                    )
                    height, width = depth.shape
                    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                    finite = (
                        np.isfinite(predicted).all(axis=-1)
                        & np.isfinite(target).all(axis=-1)
                        & np.isfinite(confidence)
                        & (predicted[..., 2] > 1e-6)
                        & (depth >= minimum_depth)
                        & (depth <= maximum_depth)
                    )
                    fit_mask = finite & (((xx + yy) % 2) == fit_parity)
                    evaluation_mask = finite & (((xx + yy) % 2) == evaluation_parity)
                    if int(fit_mask.sum()) < minimum_fit or int(evaluation_mask.sum()) < minimum_evaluation:
                        raise RuntimeError("EXP-061 has insufficient registered fit/evaluation pixels")
                    fit_source = _subsample(predicted[fit_mask], maximum_fit)
                    fit_target = _subsample(target[fit_mask], maximum_fit)
                    if fit_source.shape != fit_target.shape:
                        raise RuntimeError("EXP-061 deterministic fit subsampling diverged")
                    threshold = float(np.quantile(confidence[evaluation_mask], high_quantile))
                    high_mask = evaluation_mask & (confidence >= threshold)
                    frames.append(
                        {
                            "frame": indices[local_index],
                            "predicted": predicted,
                            "target": target,
                            "depth": depth,
                            "confidence": confidence,
                            "fit_source": fit_source,
                            "fit_target": fit_target,
                            "evaluation_mask": evaluation_mask,
                            "high_mask": high_mask,
                            "median_depth": float(np.median(depth[finite])),
                        }
                    )

            context_transform = _umeyama(
                np.concatenate([frame["fit_source"] for frame in frames], axis=0),
                np.concatenate([frame["fit_target"] for frame in frames], axis=0),
            )
            frame_transforms = [
                _umeyama(frame["fit_source"], frame["fit_target"]) for frame in frames
            ]
            current_frame_rows = []
            for local_index, (frame, frame_transform) in enumerate(
                zip(frames, frame_transforms, strict=True)
            ):
                cyclic_transform = frame_transforms[(local_index + 1) % len(frame_transforms)]
                context_epe = _relative_epe(
                    context_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["evaluation_mask"],
                )
                per_frame_epe = _relative_epe(
                    frame_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["evaluation_mask"],
                )
                cyclic_epe = _relative_epe(
                    cyclic_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["evaluation_mask"],
                )
                high_context_epe = _relative_epe(
                    context_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["high_mask"],
                )
                high_per_frame_epe = _relative_epe(
                    frame_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["high_mask"],
                )
                high_cyclic_epe = _relative_epe(
                    cyclic_transform,
                    frame["predicted"],
                    frame["target"],
                    frame["depth"],
                    frame["high_mask"],
                )
                relative_rotation = frame_transform.rotation @ context_transform.rotation.T
                row = {
                    "scene": scene,
                    "sequence": sequence,
                    "target_frame": int(target_frame),
                    "frame": int(frame["frame"]),
                    "context_epe": context_epe,
                    "per_frame_epe": per_frame_epe,
                    "cyclic_epe": cyclic_epe,
                    "context_gain": context_epe - per_frame_epe,
                    "cyclic_gain": cyclic_epe - per_frame_epe,
                    "high_context_epe": high_context_epe,
                    "high_per_frame_epe": high_per_frame_epe,
                    "high_cyclic_epe": high_cyclic_epe,
                    "high_context_gain": high_context_epe - high_per_frame_epe,
                    "fit_pixels": int(frame["fit_source"].shape[0]),
                    "evaluation_pixels": int(frame["evaluation_mask"].sum()),
                    "high_confidence_pixels": int(frame["high_mask"].sum()),
                    "mean_confidence": float(np.mean(frame["confidence"][frame["evaluation_mask"]])),
                    "mean_log_confidence": float(
                        np.mean(np.log(np.maximum(frame["confidence"][frame["evaluation_mask"]], 1e-12)))
                    ),
                    "absolute_log_scale_delta": abs(
                        math.log(frame_transform.scale / context_transform.scale)
                    ),
                    "rotation_delta_radians": _rotation_angle(relative_rotation),
                    "normalized_translation_delta": float(
                        np.linalg.norm(frame_transform.translation - context_transform.translation)
                        / max(frame["median_depth"], 1e-6)
                    ),
                }
                frame_rows.append(row)
                current_frame_rows.append(row)

            context_row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": int(target_frame),
            }
            for key in (
                "context_epe",
                "per_frame_epe",
                "cyclic_epe",
                "context_gain",
                "cyclic_gain",
                "high_context_epe",
                "high_per_frame_epe",
                "high_cyclic_epe",
                "high_context_gain",
            ):
                context_row[key] = float(np.mean([row[key] for row in current_frame_rows]))
            context_rows.append(context_row)
            print(
                f"[{len(context_rows):02d}/16] {sequence}:{target_frame} "
                f"context={context_row['context_epe']:.5f} "
                f"frame={context_row['per_frame_epe']:.5f} "
                f"cyclic={context_row['cyclic_epe']:.5f}",
                flush=True,
            )
            del state, frames
            gc.collect()
            torch.cuda.empty_cache()

    scene_context_gain = _scene_means(context_rows, "context_gain")
    scene_cyclic_gain = _scene_means(context_rows, "cyclic_gain")
    scene_high_gain = _scene_means(context_rows, "high_context_gain")
    summary = {
        key: float(np.mean([row[key] for row in context_rows]))
        for key in (
            "context_epe",
            "per_frame_epe",
            "cyclic_epe",
            "context_gain",
            "cyclic_gain",
            "high_context_epe",
            "high_per_frame_epe",
            "high_cyclic_epe",
            "high_context_gain",
        )
    }
    summary["gauge_fraction"] = summary["context_gain"] / summary["context_epe"]
    summary["high_confidence_gauge_fraction"] = (
        summary["high_context_gain"] / summary["high_context_epe"]
    )
    samples = int(config["statistics"]["bootstrap_samples"])
    intervals = {
        "context_gain": _stratified_bootstrap(
            context_rows, "context_epe", "per_frame_epe", samples, seed + 1
        ),
        "cyclic_gain": _stratified_bootstrap(
            context_rows, "cyclic_epe", "per_frame_epe", samples, seed + 2
        ),
        "high_context_gain": _stratified_bootstrap(
            context_rows, "high_context_epe", "high_per_frame_epe", samples, seed + 3
        ),
    }
    descriptive = {
        "uncertainty_vs_context_gain_spearman": _spearman(
            [-row["mean_log_confidence"] for row in frame_rows],
            [row["context_gain"] for row in frame_rows],
        ),
        "uncertainty_vs_log_scale_delta_spearman": _spearman(
            [-row["mean_log_confidence"] for row in frame_rows],
            [row["absolute_log_scale_delta"] for row in frame_rows],
        ),
        "uncertainty_vs_rotation_delta_spearman": _spearman(
            [-row["mean_log_confidence"] for row in frame_rows],
            [row["rotation_delta_radians"] for row in frame_rows],
        ),
        "uncertainty_vs_translation_delta_spearman": _spearman(
            [-row["mean_log_confidence"] for row in frame_rows],
            [row["normalized_translation_delta"] for row in frame_rows],
        ),
    }
    success = config["success"]
    gates = {
        "exact_counts": (
            len(set(row["scene"] for row in context_rows)) == int(success["exact_scenes"])
            and len(context_rows) == int(success["exact_contexts"])
            and len(frame_rows) == int(success["exact_frames"])
        ),
        "context_gain_all_scenes": all(value > 0 for value in scene_context_gain.values()),
        "cyclic_gain_all_scenes": all(value > 0 for value in scene_cyclic_gain.values()),
        "context_bootstrap_lower_positive": intervals["context_gain"][0] > 0,
        "cyclic_bootstrap_lower_positive": intervals["cyclic_gain"][0] > 0,
        "minimum_gauge_fraction": summary["gauge_fraction"]
        >= float(success["minimum_gauge_fraction"]),
        "high_confidence_gain_all_scenes": all(value > 0 for value in scene_high_gain.values()),
        "minimum_high_confidence_gauge_fraction": summary["high_confidence_gauge_fraction"]
        >= float(success["minimum_high_confidence_gauge_fraction"]),
    }
    result = {
        "experiment": "EXP-061",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "depth_preparation_sha256": _sha256(preparation_path),
        "validation_accessed": False,
        "terminal_accessed": False,
        "fit_pixel_evaluation_overlap": 0,
        "summary": summary,
        "scene_context_gain": scene_context_gain,
        "scene_cyclic_gain": scene_cyclic_gain,
        "scene_high_confidence_gain": scene_high_gain,
        "bootstrap_95": intervals,
        "descriptive": descriptive,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "context_rows": context_rows,
        "frame_rows": frame_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("summary", "bootstrap_95", "gates", "passed")}, indent=2))


if __name__ == "__main__":
    main()
