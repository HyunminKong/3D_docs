#!/usr/bin/env python3
"""Zero-fit future-revisit geometry-teacher premise for EXP-071."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _model_view,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _query_view(view: dict, index: int) -> dict:
    query = _model_view(view, index)
    query["update"] = torch.tensor([False])
    query["reset"] = torch.tensor([False])
    return query


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


def _evaluate_target(
    first: np.ndarray,
    prefix: np.ndarray,
    future: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    minimum_depth: float,
    maximum_depth: float,
    minimum_valid: int,
) -> dict:
    target = _camera_points(depth, intrinsics)
    valid = (
        np.isfinite(target).all(axis=-1)
        & (depth >= minimum_depth)
        & (depth <= maximum_depth)
    )
    for prediction in (first, prefix, future):
        valid &= np.isfinite(prediction).all(axis=-1) & (prediction[..., 2] > 1e-6)
    if int(valid.sum()) < minimum_valid:
        raise RuntimeError("EXP-071 has insufficient common metric pixels")

    aligned: dict[str, np.ndarray] = {}
    errors: dict[str, float] = {}
    for name, prediction in (
        ("first", first),
        ("prefix", prefix),
        ("future", future),
    ):
        scale = float(np.median(depth[valid]) / np.median(prediction[..., 2][valid]))
        aligned[name] = scale * prediction[valid]
        errors[name] = float(
            np.mean(
                np.linalg.norm(aligned[name] - target[valid], axis=-1)
                / np.maximum(depth[valid], minimum_depth)
            )
        )

    # Independently remove the monocular scale gauge before asking whether the
    # future-only displacement points toward the absolute metric residual.
    target_normalized = target[valid] / np.median(depth[valid])
    prefix_normalized = prefix[valid] / np.median(prefix[..., 2][valid])
    future_normalized = future[valid] / np.median(future[..., 2][valid])
    correction = future_normalized - prefix_normalized
    oracle_residual = target_normalized - prefix_normalized
    correction_flat = correction.reshape(-1)
    oracle_flat = oracle_residual.reshape(-1)
    denominator = float(
        np.linalg.norm(correction_flat) * np.linalg.norm(oracle_flat)
    )
    cosine = (
        float(np.dot(correction_flat, oracle_flat) / denominator)
        if denominator > 1e-12
        else 0.0
    )

    return {
        "valid_pixels": int(valid.sum()),
        "first_epe": errors["first"],
        "prefix_revisit_epe": errors["prefix"],
        "future_revisit_epe": errors["future"],
        "future_gain_over_first": errors["first"] - errors["future"],
        "future_gain_over_prefix": errors["prefix"] - errors["future"],
        "future_relative_gain_over_prefix": (
            errors["prefix"] - errors["future"]
        )
        / max(errors["prefix"], 1e-12),
        "future_beats_prefix": errors["future"] < errors["prefix"],
        "correction_oracle_cosine": cosine,
        "normalized_correction_rms": float(np.sqrt(np.mean(correction**2))),
    }


def _stratified_context_bootstrap(
    contexts: list[dict], samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    sequences = sorted({row["sequence"] for row in contexts})
    grouped = {
        sequence: [row for row in contexts if row["sequence"] == sequence]
        for sequence in sequences
    }
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        sequence_means = []
        for sequence in sequences:
            rows = grouped[sequence]
            indices = rng.integers(0, len(rows), size=len(rows))
            sequence_means.append(
                np.mean([rows[index]["mean_future_gain_over_prefix"] for index in indices])
            )
        draws[sample] = np.mean(sequence_means)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-071_future_revisit_teacher_premise_v10.yaml",
    )
    parser.add_argument(
        "--confirm-train-rgbd-future-revisit-premise", action="store_true"
    )
    args = parser.parse_args()
    if not args.confirm_train_rgbd_future_revisit_premise:
        raise SystemExit("EXP-071 requires explicit train-RGB-D premise confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-071 requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-071 immutable result already exists")
    manifest = json.loads(manifest_path.read_text())
    allowed_sequences = {item["relative_path"] for item in manifest["sequences"]}
    sequences = list(config["data"]["sequences"])
    starts = [int(value) for value in config["data"]["window_starts"]]
    target_offsets = [int(value) for value in config["data"]["target_offsets"]]
    window_length = int(config["data"]["window_length"])
    if not (
        manifest["role"] == "train"
        and manifest["terminal_accessed"] is False
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(sequences).issubset(allowed_sequences)
        and len(sequences) == len(set(sequences))
        and target_offsets == [3, 7, 11]
        and starts == [64, 224, 384]
        and window_length == 16
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and config["carrier"]["mode"] == "ttt3r"
    ):
        raise RuntimeError("EXP-071 source-safe frozen contract failed")

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
    carrier.eval().requires_grad_(False)

    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    minimum_valid = int(config["metric"]["minimum_valid_pixels"])
    target_rows: list[dict] = []
    context_rows: list[dict] = []
    replay_rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()

    expected_contexts = len(sequences) * len(starts)
    for sequence in sequences:
        scene = sequence.split("/", 1)[0]
        for start in starts:
            frame_indices = list(range(start, start + window_length))
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
            state = None
            first_predictions: dict[int, np.ndarray] = {}
            prefix_predictions: dict[int, np.ndarray] = {}
            with torch.no_grad():
                for offset, view in enumerate(views):
                    prediction, state, _ = carrier.step(
                        _model_view(view, offset), state
                    )
                    if offset in target_offsets:
                        first_predictions[offset] = (
                            prediction["pts3d_in_self_view"][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        prefix_prediction, _, _ = carrier.step(
                            _query_view(view, window_length + offset), state
                        )
                        prefix_predictions[offset] = (
                            prefix_prediction["pts3d_in_self_view"][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                final_state = state
                future_predictions: dict[int, np.ndarray] = {}
                for offset in target_offsets:
                    future_prediction, _, _ = carrier.step(
                        _query_view(views[offset], window_length + offset),
                        final_state,
                    )
                    future_predictions[offset] = (
                        future_prediction["pts3d_in_self_view"][0]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                replay_offset = target_offsets[len(target_offsets) // 2]
                replay_prediction, _, _ = carrier.step(
                    _query_view(
                        views[replay_offset], window_length + replay_offset
                    ),
                    final_state,
                )
                replay = (
                    replay_prediction["pts3d_in_self_view"][0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            replay_difference = float(
                np.max(np.abs(future_predictions[replay_offset] - replay))
            )
            replay_rows.append(
                {
                    "scene": scene,
                    "sequence": sequence,
                    "window_start": start,
                    "target_offset": replay_offset,
                    "maximum_abs_point_difference": replay_difference,
                }
            )

            context_target_rows = []
            for offset in target_offsets:
                depth = np.asarray(views[offset]["depthmap"], dtype=np.float64)
                intrinsics = np.asarray(
                    views[offset]["camera_intrinsics"], dtype=np.float64
                )
                metrics = _evaluate_target(
                    first_predictions[offset],
                    prefix_predictions[offset],
                    future_predictions[offset],
                    depth,
                    intrinsics,
                    minimum_depth,
                    maximum_depth,
                    minimum_valid,
                )
                row = {
                    "scene": scene,
                    "sequence": sequence,
                    "window_start": start,
                    "target_offset": offset,
                    "target_frame": start + offset,
                    "future_frames_after_target": window_length - offset - 1,
                    "query_rgb_identical": True,
                    "prefix_query_update": False,
                    "future_query_update": False,
                    **metrics,
                }
                target_rows.append(row)
                context_target_rows.append(row)

            context_row = {
                "scene": scene,
                "sequence": sequence,
                "window_start": start,
                "targets": len(context_target_rows),
                "mean_future_gain_over_prefix": float(
                    np.mean(
                        [
                            row["future_gain_over_prefix"]
                            for row in context_target_rows
                        ]
                    )
                ),
                "mean_correction_oracle_cosine": float(
                    np.mean(
                        [
                            row["correction_oracle_cosine"]
                            for row in context_target_rows
                        ]
                    )
                ),
                "replay_maximum_abs_difference": replay_difference,
            }
            context_rows.append(context_row)
            print(
                f"[{len(context_rows):02d}/{expected_contexts}] "
                f"{sequence}:{start} gain="
                f"{context_row['mean_future_gain_over_prefix']:.6f} "
                f"cos={context_row['mean_correction_oracle_cosine']:.3f}",
                flush=True,
            )
            del (
                views,
                first_predictions,
                prefix_predictions,
                future_predictions,
                replay,
                final_state,
            )
            gc.collect()
            torch.cuda.empty_cache()

    scene_gains = {
        scene: float(
            np.mean(
                [
                    row["future_gain_over_prefix"]
                    for row in target_rows
                    if row["scene"] == scene
                ]
            )
        )
        for scene in sorted({row["scene"] for row in target_rows})
    }
    mean_first = float(np.mean([row["first_epe"] for row in target_rows]))
    mean_prefix = float(
        np.mean([row["prefix_revisit_epe"] for row in target_rows])
    )
    mean_future = float(
        np.mean([row["future_revisit_epe"] for row in target_rows])
    )
    mean_gain_prefix = mean_prefix - mean_future
    relative_gain_prefix = mean_gain_prefix / max(mean_prefix, 1e-12)
    target_win_fraction = float(
        np.mean([row["future_beats_prefix"] for row in target_rows])
    )
    mean_cosine = float(
        np.mean([row["correction_oracle_cosine"] for row in target_rows])
    )
    interval = _stratified_context_bootstrap(
        context_rows,
        int(config["statistics"]["bootstrap_samples"]),
        seed + 1,
    )
    maximum_replay = max(
        row["maximum_abs_point_difference"] for row in replay_rows
    )
    success = config["success"]
    gates = {
        "exact_counts": (
            len(scene_gains) == int(success["exact_scenes"])
            and len({row["sequence"] for row in target_rows})
            == int(success["exact_sequences"])
            and len(context_rows) == int(success["exact_contexts"])
            and len(target_rows) == int(success["exact_targets"])
            and len(replay_rows) == int(success["exact_replay_checks"])
        ),
        "replay_within_tolerance": maximum_replay
        <= float(success["maximum_replay_abs_difference"]),
        "future_gain_over_prefix_positive_all_scenes": all(
            value > 0 for value in scene_gains.values()
        ),
        "bootstrap_lower_positive": interval[0] > 0,
        "minimum_relative_future_over_prefix_gain": relative_gain_prefix
        >= float(success["minimum_relative_future_over_prefix_gain"]),
        "minimum_target_win_fraction": target_win_fraction
        >= float(success["minimum_target_win_fraction"]),
        "minimum_correction_oracle_cosine": mean_cosine
        >= float(success["minimum_correction_oracle_cosine"]),
    }
    result = {
        "experiment": "EXP-071",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "train_manifest_sha256": _sha256(manifest_path),
        "validation_accessed": False,
        "terminal_accessed": False,
        "model_fitted": False,
        "query_rgb_identical": True,
        "query_updates_disabled": True,
        "future_teacher_role": "offline_premise_only",
        "summary": {
            "mean_first_epe": mean_first,
            "mean_prefix_revisit_epe": mean_prefix,
            "mean_future_revisit_epe": mean_future,
            "mean_future_gain_over_first": mean_first - mean_future,
            "mean_future_gain_over_prefix": mean_gain_prefix,
            "relative_future_gain_over_prefix": relative_gain_prefix,
            "target_win_fraction_over_prefix": target_win_fraction,
            "mean_correction_oracle_cosine": mean_cosine,
            "maximum_replay_abs_difference": maximum_replay,
        },
        "scene_future_gains_over_prefix": scene_gains,
        "future_gain_over_prefix_bootstrap_95": interval,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "context_rows": context_rows,
        "target_rows": target_rows,
        "replay_rows": replay_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "summary",
                    "scene_future_gains_over_prefix",
                    "future_gain_over_prefix_bootstrap_95",
                    "gates",
                    "passed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
