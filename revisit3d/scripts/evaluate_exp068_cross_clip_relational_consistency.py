#!/usr/bin/env python3
"""Evaluate frozen cross-clip query protocols without fitting a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base: int, text: str) -> int:
    token = hashlib.sha256(f"{base}::{text}".encode()).digest()
    return int.from_bytes(token[:8], "little") % (2**32 - 1)


def fit_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * R @ src + translation with Umeyama."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[valid], dst[valid]
    if src.shape[0] < 3:
        raise ValueError("At least three finite points are required for Sim(3)")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    covariance = (dst_c.T @ src_c) / float(src.shape[0])
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.square(src_c).sum() / src.shape[0])
    scale = float((singular * sign).sum() / max(variance, 1e-12))
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_sim3(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    points = np.asarray(points, dtype=np.float64)
    return (scale * (rotation @ points.T)).T + translation


def fit_layer_sim3(
    src_cal: np.ndarray,
    dst_cal: np.ndarray,
    reference_depth_cal: np.ndarray,
    layers: int,
    minimum_points: int,
) -> tuple[np.ndarray, list[tuple[float, np.ndarray, np.ndarray]], int]:
    finite_depth = reference_depth_cal[np.isfinite(reference_depth_cal)]
    if finite_depth.size < minimum_points:
        raise ValueError("Insufficient reference depths for layer alignment")
    boundaries = np.quantile(finite_depth, np.linspace(0.0, 1.0, layers + 1)[1:-1])
    global_transform = fit_sim3(src_cal, dst_cal)
    transforms: list[tuple[float, np.ndarray, np.ndarray]] = []
    fallbacks = 0
    bins = np.digitize(reference_depth_cal, boundaries, right=False)
    for layer in range(layers):
        selected = bins == layer
        if int(selected.sum()) >= max(3, minimum_points // layers):
            transforms.append(fit_sim3(src_cal[selected], dst_cal[selected]))
        else:
            transforms.append(global_transform)
            fallbacks += 1
    return boundaries, transforms, fallbacks


def apply_layer_sim3(
    points: np.ndarray,
    reference_depth: np.ndarray,
    boundaries: np.ndarray,
    transforms: list[tuple[float, np.ndarray, np.ndarray]],
) -> np.ndarray:
    output = np.empty_like(np.asarray(points, dtype=np.float64))
    bins = np.digitize(reference_depth, boundaries, right=False)
    for layer, transform in enumerate(transforms):
        selected = bins == layer
        output[selected] = apply_sim3(points[selected], transform)
    return output


def aligned_point_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    scene_scale: float,
    thresholds: list[float],
) -> tuple[float, float]:
    valid = np.isfinite(prediction).all(axis=1) & np.isfinite(truth).all(axis=1)
    prediction, truth = prediction[valid], truth[valid]
    pred_norm = np.linalg.norm(prediction, axis=1)
    truth_norm = np.linalg.norm(truth, axis=1)
    scale = float(np.median(truth_norm) / max(float(np.median(pred_norm)), 1e-12))
    error = np.linalg.norm(prediction * scale - truth, axis=1)
    apd = float(np.mean([np.mean(error < fraction * scene_scale) for fraction in thresholds]))
    return float(error.mean()), apd


def pair_disagreement(
    reference: np.ndarray,
    candidate: np.ndarray,
    scene_scale: float,
    knn: int,
    max_distance_fraction: float,
) -> tuple[float, int]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    valid = np.isfinite(reference).all(axis=1) & np.isfinite(candidate).all(axis=1)
    reference, candidate = reference[valid], candidate[valid]
    if reference.shape[0] < knn + 1:
        return float("nan"), 0
    distances = np.linalg.norm(reference[:, None] - reference[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argpartition(distances, kth=min(knn, reference.shape[0] - 1), axis=1)[:, :knn]
    pairs: set[tuple[int, int]] = set()
    maximum = max_distance_fraction * scene_scale
    for i, row in enumerate(neighbors):
        for j in row:
            a, b = sorted((int(i), int(j)))
            if distances[a, b] <= maximum:
                pairs.add((a, b))
    if not pairs:
        return float("nan"), 0
    ids_a = np.fromiter((pair[0] for pair in pairs), dtype=np.int64)
    ids_b = np.fromiter((pair[1] for pair in pairs), dtype=np.int64)
    ref_distance = np.linalg.norm(reference[ids_a] - reference[ids_b], axis=1)
    cand_distance = np.linalg.norm(candidate[ids_a] - candidate[ids_b], axis=1)
    return float(np.mean(np.abs(ref_distance - cand_distance)) / scene_scale), len(pairs)


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def mean_rows(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks for ties, starting at zero."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    ranks_a = average_ranks(np.asarray(values_a, dtype=np.float64))
    ranks_b = average_ranks(np.asarray(values_b, dtype=np.float64))
    if ranks_a.size < 2 or np.std(ranks_a) <= 0.0 or np.std(ranks_b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-068_cross_clip_relational_consistency_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    experiment = str(config.get("experiment", "EXP-068"))
    role = str(config["data"].get("role", "premise"))
    manifest_path = Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    output_path = Path(config["output"]["result"])
    if output_path.exists():
        raise RuntimeError(f"{experiment} result already exists: {output_path}")
    if not (
        manifest["experiment"] == "EXP-068"
        and manifest["npz_content_accessed"] is False
        and manifest["model_accessed"] is False
        and len(manifest["roles"]["premise"]) == 16
        and len(manifest["roles"]["validation"]) == 11
        and len(manifest["roles"]["terminal"]) == 12
        and config["data"]["terminal_access"] is False
        and role in {"premise", "validation"}
    ):
        raise RuntimeError(f"{experiment} source-safe role contract failed")
    if experiment == "EXP-068" and config["data"]["validation_access"] is not False:
        raise RuntimeError("EXP-068 validation role must remain unopened")
    if experiment == "EXP-069" and not (
        role == "validation"
        and config["data"]["role_content_accessed_before_registration"] is False
    ):
        raise RuntimeError("EXP-069 must use the previously unopened validation role")
    role_entries = manifest["roles"][role]

    external_root = Path(config["carrier"]["repository"]).resolve()
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from eval_track3d_in_worldtrack import load_worldtrack_sequence
    from infer_track_3d import _resize_video, _resolve_device, _unwrap_state_dict
    from src.core import load_checkpoint, load_yaml_config, seed_everything
    from src.eval.tasks import _encode_model_memory, _model_clip_frames, _run_model_for_queries
    from src.model import build_model

    device = _resolve_device("cuda")
    model_config_path = Path(config["carrier"]["model_config"])
    checkpoint_path = Path(config["carrier"]["checkpoint"])
    model_config = load_yaml_config(model_config_path)
    seed_everything(int(config["seed"]), deterministic=True)
    model = build_model(model_config["model"]).eval()
    state = _unwrap_state_dict(load_checkpoint(checkpoint_path, map_location="cpu"))
    load_result = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    if _model_clip_frames(model) != int(config["carrier"]["clip_frames"]):
        raise RuntimeError(f"{experiment} checkpoint clip length does not match the frozen config")

    image_hw = tuple(int(value) for value in config["carrier"]["image_size"])
    clip_specs = config["query"]["clips"]
    source_global = int(config["query"]["source_global_frame"])
    targets_global = [int(value) for value in config["query"]["target_global_frames"]]
    maximum_tracks = int(config["query"]["maximum_tracks_per_sequence"])
    split_seed = int(config["query"]["split_seed"])
    chunk_size = int(config["query"]["chunk_size"])
    layers = int(config["alignment"]["depth_layer"]["layers"])
    minimum_alignment = int(config["alignment"]["minimum_alignment_points"])
    thresholds = [float(value) for value in config["metric"]["apd_relative_thresholds"]]
    knn = int(config["metric"]["pair_knn"])
    pair_max = float(config["metric"]["pair_max_distance_scene_fraction"])
    data_root = Path(config["data"]["root"])

    sequence_rows: list[dict[str, Any]] = []
    attempted_sequences: list[str] = []
    coverage_exclusions: list[dict[str, Any]] = []
    replay_max = 0.0
    total_layer_fallbacks = 0
    for sequence_entry in role_entries:
        sequence_name = sequence_entry["sequence"]
        attempted_sequences.append(sequence_name)
        sequence_path = data_root / sequence_entry["relative_path"]
        sample = load_worldtrack_sequence(sequence_path, num_frames=48)
        video_rgb = np.asarray(sample["video_rgb"][:48])
        gt_cam = np.asarray(sample["tracks_xyz_cam"], dtype=np.float64)[:48]
        gt_uv = np.asarray(sample["tracks_uv"], dtype=np.float64)[:48]
        visibility = np.asarray(sample["visibility"], dtype=bool)[:48]
        if video_rgb.shape[0] != 48:
            raise RuntimeError(f"{experiment} {sequence_name} has fewer than 48 frames")
        original_h, original_w = int(video_rgb.shape[1]), int(video_rgb.shape[2])

        valid_tracks = visibility[source_global].copy()
        valid_tracks &= np.isfinite(gt_uv[source_global]).all(axis=1)
        for target in targets_global:
            valid_tracks &= np.isfinite(gt_cam[target]).all(axis=1)
        track_ids = np.flatnonzero(valid_tracks)
        if track_ids.size < minimum_alignment * 2:
            if experiment == "EXP-069" and config["protocol_revision"] == "v1.1":
                coverage_exclusions.append(
                    {
                        "sequence": sequence_name,
                        "reason": "insufficient_eligible_tracks",
                        "eligible_tracks": int(track_ids.size),
                        "required_tracks": int(minimum_alignment * 2),
                    }
                )
                print(
                    f"[{experiment}] {sequence_name}: coverage exclusion, "
                    f"eligible={track_ids.size}, required={minimum_alignment * 2}"
                )
                continue
            raise RuntimeError(f"{experiment} {sequence_name} has insufficient finite tracks")
        if track_ids.size > maximum_tracks:
            rng_select = np.random.default_rng(stable_seed(split_seed, sequence_name + "::select"))
            track_ids = np.sort(rng_select.choice(track_ids, size=maximum_tracks, replace=False))
        rng_split = np.random.default_rng(stable_seed(split_seed, sequence_name + "::split"))
        shuffled = track_ids.copy()
        rng_split.shuffle(shuffled)
        split = int(round(shuffled.size * float(config["query"]["alignment_fraction"])))
        alignment_ids = np.sort(shuffled[:split])
        evaluation_ids = np.sort(shuffled[split:])
        if min(alignment_ids.size, evaluation_ids.size) < minimum_alignment:
            raise RuntimeError(f"{experiment} {sequence_name} alignment/evaluation split is too small")

        uv = gt_uv[source_global, track_ids].astype(np.float32)
        uv[:, 0] /= float(max(original_w - 1, 1))
        uv[:, 1] /= float(max(original_h - 1, 1))
        uv = np.clip(uv, 0.0, 1.0)
        id_to_local = {int(track): index for index, track in enumerate(track_ids)}
        cal_local = np.asarray([id_to_local[int(track)] for track in alignment_ids], dtype=np.int64)
        eval_local = np.asarray([id_to_local[int(track)] for track in evaluation_ids], dtype=np.int64)

        memories: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for clip_name, spec in clip_specs.items():
            start, end = int(spec["start"]), int(spec["end"])
            clip_rgb = video_rgb[start:end]
            if clip_rgb.shape[0] != 32:
                raise RuntimeError("EXP-068 clip does not contain exactly 32 frames")
            resized = _resize_video(clip_rgb, image_hw=image_hw)
            video = (
                torch.from_numpy(resized)
                .to(device, torch.float32)
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                / 255.0
            )
            aspect = torch.tensor([[original_w / original_h]], device=device, dtype=torch.float32)
            memory = _encode_model_memory(model=model, video_b=video, aspect_b=aspect)
            memories[clip_name] = (video, aspect, memory)

        target_rows: list[dict[str, Any]] = []
        for target_global in targets_global:
            predictions: dict[str, np.ndarray] = {}
            for clip_name, spec in clip_specs.items():
                start = int(spec["start"])
                video, aspect, memory = memories[clip_name]
                query = {
                    "u": torch.from_numpy(uv[:, 0]).to(device),
                    "v": torch.from_numpy(uv[:, 1]).to(device),
                    "t_src": torch.full(
                        (track_ids.size,), source_global - start, device=device, dtype=torch.long
                    ),
                    "t_tgt": torch.full(
                        (track_ids.size,), target_global - start, device=device, dtype=torch.long
                    ),
                    "t_cam": torch.full(
                        (track_ids.size,), target_global - start, device=device, dtype=torch.long
                    ),
                }
                output = _run_model_for_queries(
                    model=model,
                    video_b=video,
                    aspect_b=aspect,
                    query=query,
                    chunk_size=chunk_size,
                    memory_b=memory,
                )
                predictions[clip_name] = output["xyz_3d"].numpy().astype(np.float64)
                if clip_name == "reference":
                    replay = _run_model_for_queries(
                        model=model,
                        video_b=video,
                        aspect_b=aspect,
                        query=query,
                        chunk_size=chunk_size,
                        memory_b=memory,
                    )["xyz_3d"].numpy().astype(np.float64)
                    replay_max = max(replay_max, float(np.max(np.abs(replay - predictions[clip_name]))))

            reference = predictions["reference"]
            large = predictions["large_shift"]
            adjacent = predictions["adjacent_shift"]
            truth = gt_cam[target_global, track_ids]
            scene_scale = float(np.median(np.linalg.norm(truth[eval_local], axis=1)))
            if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
                raise RuntimeError(f"{experiment} invalid scene scale for {sequence_name}/{target_global}")

            large_global_transform = fit_sim3(large[cal_local], reference[cal_local])
            adjacent_global_transform = fit_sim3(adjacent[cal_local], reference[cal_local])
            large_global = apply_sim3(large[eval_local], large_global_transform)
            adjacent_global = apply_sim3(adjacent[eval_local], adjacent_global_transform)
            reference_eval = reference[eval_local]

            ref_depth_cal = np.linalg.norm(reference[cal_local], axis=1)
            ref_depth_eval = np.linalg.norm(reference_eval, axis=1)
            boundaries, large_layers, large_fallbacks = fit_layer_sim3(
                large[cal_local], reference[cal_local], ref_depth_cal, layers, minimum_alignment
            )
            _, adjacent_layers, adjacent_fallbacks = fit_layer_sim3(
                adjacent[cal_local], reference[cal_local], ref_depth_cal, layers, minimum_alignment
            )
            total_layer_fallbacks += large_fallbacks + adjacent_fallbacks
            large_layer = apply_layer_sim3(large[eval_local], ref_depth_eval, boundaries, large_layers)
            adjacent_layer = apply_layer_sim3(
                adjacent[eval_local], ref_depth_eval, boundaries, adjacent_layers
            )

            raw_large = float(np.linalg.norm(reference_eval - large[eval_local], axis=1).mean() / scene_scale)
            raw_adjacent = float(
                np.linalg.norm(reference_eval - adjacent[eval_local], axis=1).mean() / scene_scale
            )
            global_large = float(np.linalg.norm(reference_eval - large_global, axis=1).mean() / scene_scale)
            global_adjacent = float(
                np.linalg.norm(reference_eval - adjacent_global, axis=1).mean() / scene_scale
            )
            layer_large = float(np.linalg.norm(reference_eval - large_layer, axis=1).mean() / scene_scale)
            layer_adjacent = float(
                np.linalg.norm(reference_eval - adjacent_layer, axis=1).mean() / scene_scale
            )
            pair_large, pair_count = pair_disagreement(
                reference_eval, large_layer, scene_scale, knn, pair_max
            )
            pair_adjacent, _ = pair_disagreement(
                reference_eval, adjacent_layer, scene_scale, knn, pair_max
            )
            epe_reference, apd_reference = aligned_point_metrics(
                reference_eval, truth[eval_local], scene_scale, thresholds
            )
            epe_large, apd_large = aligned_point_metrics(
                large[eval_local], truth[eval_local], scene_scale, thresholds
            )
            target_rows.append(
                {
                    "target_global_frame": target_global,
                    "tracks_alignment": int(cal_local.size),
                    "tracks_evaluation": int(eval_local.size),
                    "scene_scale": scene_scale,
                    "raw_large_fraction": raw_large,
                    "raw_adjacent_fraction": raw_adjacent,
                    "global_large_fraction": global_large,
                    "global_adjacent_fraction": global_adjacent,
                    "layer_large_fraction": layer_large,
                    "layer_adjacent_fraction": layer_adjacent,
                    "layer_retention": layer_large / max(raw_large, 1e-12),
                    "pair_large_fraction": pair_large,
                    "pair_adjacent_fraction": pair_adjacent,
                    "pair_count": pair_count,
                    "epe_reference": epe_reference,
                    "epe_large": epe_large,
                    "apd_reference": apd_reference,
                    "apd_large": apd_large,
                    "signed_apd_gain": apd_large - apd_reference,
                    "absolute_apd_difference": abs(apd_reference - apd_large),
                    "large_layer_fallbacks": large_fallbacks,
                    "adjacent_layer_fallbacks": adjacent_fallbacks,
                }
            )

        sequence_rows.append(
            {
                "sequence": sequence_name,
                "family": sequence_entry["filename_family"],
                "targets": target_rows,
                "mean_raw_large_fraction": mean_rows(target_rows, "raw_large_fraction"),
                "mean_global_large_fraction": mean_rows(target_rows, "global_large_fraction"),
                "mean_layer_large_fraction": mean_rows(target_rows, "layer_large_fraction"),
                "mean_layer_adjacent_fraction": mean_rows(target_rows, "layer_adjacent_fraction"),
                "mean_large_minus_adjacent_fraction": mean_rows(
                    target_rows, "layer_large_fraction"
                )
                - mean_rows(target_rows, "layer_adjacent_fraction"),
                "mean_layer_retention": mean_rows(target_rows, "layer_retention"),
                "mean_pair_large_fraction": mean_rows(target_rows, "pair_large_fraction"),
                "mean_absolute_apd_difference": mean_rows(
                    target_rows, "absolute_apd_difference"
                ),
                "mean_signed_apd_gain": mean_rows(target_rows, "signed_apd_gain"),
            }
        )
        print(
            f"[{experiment}] {sequence_name}: layer-large="
            f"{sequence_rows[-1]['mean_layer_large_fraction']:.6f}, "
            f"large-adj={sequence_rows[-1]['mean_large_minus_adjacent_fraction']:.6f}"
        )
        del memories
        torch.cuda.empty_cache()

    samples = int(config["statistics"]["bootstrap_samples"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    layer_values = [row["mean_layer_large_fraction"] for row in sequence_rows]
    delta_values = [row["mean_large_minus_adjacent_fraction"] for row in sequence_rows]
    pair_values = [row["mean_pair_large_fraction"] for row in sequence_rows]
    retention_values = [row["mean_layer_retention"] for row in sequence_rows]
    apd_values = [row["mean_absolute_apd_difference"] for row in sequence_rows]
    signed_apd_values = [row["mean_signed_apd_gain"] for row in sequence_rows]
    target_delta_values = [
        float(target["layer_large_fraction"] - target["layer_adjacent_fraction"])
        for row in sequence_rows
        for target in row["targets"]
    ]
    target_signed_apd_values = [
        float(target["signed_apd_gain"])
        for row in sequence_rows
        for target in row["targets"]
    ]
    aggregate = {
        "attempted_sequences": len(attempted_sequences),
        "coverage_exclusions": len(coverage_exclusions),
        "sequences": len(sequence_rows),
        "targets": sum(len(row["targets"]) for row in sequence_rows),
        "maximum_replay_abs_difference": replay_max,
        "mean_raw_large_fraction": float(
            np.mean([row["mean_raw_large_fraction"] for row in sequence_rows])
        ),
        "mean_global_large_fraction": float(
            np.mean([row["mean_global_large_fraction"] for row in sequence_rows])
        ),
        "mean_layer_large_fraction": float(np.mean(layer_values)),
        "layer_large_ci95": bootstrap_ci(layer_values, samples, bootstrap_seed),
        "mean_layer_adjacent_fraction": float(
            np.mean([row["mean_layer_adjacent_fraction"] for row in sequence_rows])
        ),
        "mean_large_minus_adjacent_fraction": float(np.mean(delta_values)),
        "large_minus_adjacent_ci95": bootstrap_ci(delta_values, samples, bootstrap_seed + 1),
        "large_over_adjacent_positive_sequences": int(np.sum(np.asarray(delta_values) > 0.0)),
        "mean_layer_retention": float(np.mean(retention_values)),
        "mean_pair_large_fraction": float(np.mean(pair_values)),
        "pair_large_ci95": bootstrap_ci(pair_values, samples, bootstrap_seed + 2),
        "mean_absolute_apd_difference": float(np.mean(apd_values)),
        "mean_signed_apd_gain": float(np.mean(signed_apd_values)),
        "signed_apd_gain_ci95": bootstrap_ci(
            signed_apd_values, samples, bootstrap_seed + 3
        ),
        "apd_positive_sequences": int(np.sum(np.asarray(signed_apd_values) > 0.0)),
        "apd_positive_targets": int(np.sum(np.asarray(target_signed_apd_values) > 0.0)),
        "query_integrity_ranking_inversions": int(
            np.sum(
                (np.asarray(target_delta_values) > 0.0)
                & (np.asarray(target_signed_apd_values) > 0.0)
            )
        ),
        "target_structural_apd_spearman": spearman(
            target_delta_values, target_signed_apd_values
        ),
        "layer_fit_fallbacks": total_layer_fallbacks,
    }

    success = config["success"]
    if experiment == "EXP-069":
        corrected_coverage = config["protocol_revision"] == "v1.1"
        expected_attempts = int(
            success["exact_attempted_sequences"] if corrected_coverage else success["exact_sequences"]
        )
        minimum_evaluable = int(
            success["minimum_evaluable_sequences"] if corrected_coverage else expected_attempts
        )
        structural_positive_minimum = (
            int(np.ceil(float(success["minimum_large_over_adjacent_positive_fraction"]) * len(sequence_rows)))
            if corrected_coverage
            else int(success["minimum_large_over_adjacent_positive_sequences"])
        )
        layer_positive_minimum = (
            int(np.ceil(float(success["minimum_layer_positive_fraction"]) * len(sequence_rows)))
            if corrected_coverage
            else int(success["minimum_layer_positive_sequences"])
        )
        apd_sequence_minimum = (
            int(np.ceil(float(success["minimum_apd_positive_sequence_fraction"]) * len(sequence_rows)))
            if corrected_coverage
            else int(success["minimum_apd_positive_sequences"])
        )
        apd_target_minimum = (
            int(np.ceil(float(success["minimum_apd_positive_target_fraction"]) * aggregate["targets"]))
            if corrected_coverage
            else int(success["minimum_apd_positive_targets"])
        )
        gates = {
            "exact_replay": replay_max <= float(success["maximum_replay_abs_difference"]),
            "attempted_all_fixed_sequences": len(attempted_sequences) == expected_attempts,
            "minimum_evaluable_sequences": len(sequence_rows) >= minimum_evaluable,
            "complete_targets": aggregate["targets"]
            == len(sequence_rows) * len(targets_global),
            "layer_residual_ci": aggregate["layer_large_ci95"][0] > 0.0,
            "layer_residual_frequency": int(np.sum(np.asarray(layer_values) > 0.0))
            >= layer_positive_minimum,
            "large_over_adjacent_ci": aggregate["large_minus_adjacent_ci95"][0] > 0.0,
            "large_over_adjacent_frequency": aggregate[
                "large_over_adjacent_positive_sequences"
            ]
            >= structural_positive_minimum,
            "mean_signed_apd_gain": aggregate["mean_signed_apd_gain"]
            >= float(success["minimum_mean_signed_apd_gain"]),
            "signed_apd_gain_ci": aggregate["signed_apd_gain_ci95"][0]
            > float(success["minimum_signed_apd_gain_ci_lower"]),
            "apd_positive_sequences": aggregate["apd_positive_sequences"]
            >= apd_sequence_minimum,
            "apd_positive_targets": aggregate["apd_positive_targets"]
            >= apd_target_minimum,
            "structural_apd_decoupling": abs(
                aggregate["target_structural_apd_spearman"]
            )
            <= float(success["maximum_absolute_target_spearman"]),
            "no_layer_fit_fallbacks": total_layer_fallbacks == 0,
            "source_safe_reassignment": role == "validation",
            "no_model_fit": True,
            "terminal_not_accessed": True,
        }
    else:
        gates = {
            "exact_replay": replay_max <= float(success["maximum_replay_abs_difference"]),
            "complete_sequences": len(sequence_rows) == int(success["exact_premise_sequences"]),
            "complete_targets": aggregate["targets"]
            == int(success["exact_premise_sequences"]) * len(targets_global),
            "layer_residual_magnitude": aggregate["mean_layer_large_fraction"]
            >= float(success["minimum_layer_residual_scene_fraction"]),
            "layer_residual_ci": aggregate["layer_large_ci95"][0] > 0.0,
            "layer_residual_all_sequences": all(value > 0.0 for value in layer_values),
            "large_over_adjacent_ci": aggregate["large_minus_adjacent_ci95"][0] > 0.0,
            "large_over_adjacent_frequency": aggregate["large_over_adjacent_positive_sequences"]
            >= int(success["minimum_large_over_adjacent_positive_sequences"]),
            "layer_residual_retention": aggregate["mean_layer_retention"]
            >= float(success["minimum_layer_residual_retention"]),
            "pair_residual_ci": aggregate["pair_large_ci95"][0] > 0.0,
            "pair_over_replay": aggregate["mean_pair_large_fraction"] > 2.0 * replay_max,
            "pointwise_apd_blindness": aggregate["mean_absolute_apd_difference"]
            < float(success["maximum_mean_absolute_apd_difference"]),
            "no_layer_fit_fallbacks": total_layer_fallbacks == 0,
            "source_safe_roles": True,
            "no_model_fit": True,
            "validation_not_accessed": True,
            "terminal_not_accessed": True,
        }
    result = {
        "experiment": experiment,
        "protocol_revision": config["protocol_revision"],
        "manifest_role_opened": role,
        "status": "passed" if all(gates.values()) else "failed",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_config_sha256": sha256_file(model_config_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "load_missing_keys": list(load_result.missing_keys),
        "load_unexpected_keys": list(load_result.unexpected_keys),
        "aggregate": aggregate,
        "gates": gates,
        "passed_gates": int(sum(gates.values())),
        "total_gates": len(gates),
        "attempted_sequence_names": attempted_sequences,
        "coverage_exclusions": coverage_exclusions,
        "sequence_rows": sequence_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": aggregate, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
