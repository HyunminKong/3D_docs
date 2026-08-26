#!/usr/bin/env python3
"""Scene-OOF pairwise-geometry observability diagnostic for EXP-056."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
)
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _model_view,
    _relative_point_loss,
    _scene_means,
)
from revisit3d.scripts.evaluate_exp054_conditional_tangent_oracle import (
    _bootstrap_mean,
    _policy_result,
    _prediction_difference,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


FEATURE_SETS = ("token", "geometry", "combined")
POLICIES = ("global", "token", "geometry", "combined", "shuffled_geometry", "oracle")


def _features(
    tokens: torch.Tensor,
    current_points: torch.Tensor,
    previous_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape[:2] != current_points.shape[:2]:
        raise ValueError("token and current-point layouts differ")
    differences = current_points[:, :, None, :] - previous_points[:, None, :, :]
    distances = torch.linalg.vector_norm(differences, dim=-1)
    nearest_distance, nearest = distances.min(dim=-1)
    nearest_previous = torch.gather(
        previous_points,
        1,
        nearest[..., None].expand(-1, -1, previous_points.shape[-1]),
    )
    residual = current_points - nearest_previous
    median = nearest_distance.median(dim=1, keepdim=True).values.clamp_min(1e-6)
    geometry = torch.cat(
        (
            residual / median[..., None],
            torch.log1p(nearest_distance / median)[..., None],
        ),
        dim=-1,
    )
    token = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    return token, geometry.float()


def _feature(record: dict, name: str) -> np.ndarray:
    if name == "token":
        return record["token"]
    if name == "geometry":
        return record["geometry"]
    if name == "combined":
        return np.concatenate((record["token"], record["geometry"]), axis=-1)
    raise KeyError(name)


def _fit_moment(records: list[dict], feature_name: str) -> dict[str, np.ndarray]:
    features = np.concatenate([_feature(record, feature_name) for record in records], axis=0)
    labels = np.concatenate([record["labels"] for record in records], axis=0)
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    standardized = (features - mean) / std
    signs = labels.astype(np.float32) * 2.0 - 1.0
    weight = (standardized.T @ signs) / np.float32(len(standardized))
    train_score = standardized @ weight
    score_rms = np.sqrt(np.mean(np.square(train_score), axis=0))
    score_rms = np.maximum(score_rms, np.float32(1e-6))
    return {"mean": mean, "std": std, "weight": weight, "score_rms": score_rms}


def _score(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["mean"]) / model["std"]
    return (standardized @ model["weight"]) / model["score_rms"]


def _balanced_accuracy(score: np.ndarray, labels: np.ndarray) -> float:
    prediction = score > 0
    labels = labels.astype(bool)
    positive = float(np.mean(prediction[labels])) if labels.any() else 0.0
    negative = float(np.mean(~prediction[~labels])) if (~labels).any() else 0.0
    return 0.5 * (positive + negative)


def _dataset_item(SevenScenes, config: dict, sequence: str, target_frame: int):
    context = int(config["data"]["context_frames"])
    indices = list(range(target_frame - context + 1, target_frame + 1))
    spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
    dataset = SevenScenes(
        split="train",
        ROOT=config["data"]["root"],
        resolution=tuple(config["carrier"]["resolution"]),
        tuple_list=[spec],
        seed=int(config["seed"]),
    )
    return dataset, dataset[0]


def _cache(carrier, gt_views, patch_size: int):
    state = None
    previous = current = auxiliary = None
    with torch.no_grad():
        for index, gt_view in enumerate(gt_views):
            prediction, state, current_auxiliary = carrier.step(
                _model_view(gt_view, index), state
            )
            if index == len(gt_views) - 2:
                previous = prediction
            if index == len(gt_views) - 1:
                current = prediction
                auxiliary = current_auxiliary
    assert previous is not None and current is not None and auxiliary is not None
    previous_points = patch_center_points(
        previous["pts3d_in_other_view"], patch_size
    ).detach()
    return current, auxiliary, previous_points


def _metric(prediction, gt_view, config):
    return _relative_point_loss(
        prediction["pts3d_in_self_view"],
        gt_view["depthmap"],
        gt_view["camera_intrinsics"],
        minimum_depth=float(config["metric"]["minimum_depth_m"]),
        maximum_depth=float(config["metric"]["maximum_depth_m"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-056_pairwise_geometry_observability_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-oof", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_oof or not torch.cuda.is_available():
        raise SystemExit("EXP-056 requires train-RGB-D OOF confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    depth_path = Path(config["data"]["depth_preparation"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-056 result already exists")
    manifest = json.loads(manifest_path.read_text())
    depth_preparation = json.loads(depth_path.read_text())
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and depth_preparation["validation_accessed"] is False
        and depth_preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-056 source-safe contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "src"))
    from eval.mv_recon.data import SevenScenes

    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
        update_type="ttt3r",
    ).cuda()
    carrier.eval()
    carrier.model.requires_grad_(False)
    carrier.residual.requires_grad_(False)
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    records = []
    torch.cuda.reset_peak_memory_stats()

    # Phase 1: create source-safe features and offline labels once.
    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame in config["data"]["target_frames"]:
            dataset, gt_views = _dataset_item(
                SevenScenes, config, sequence, int(target_frame)
            )
            base, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
            tokens = auxiliary["decoder_patch_tokens"].float()
            zero = torch.zeros(
                1, tokens.shape[1], carrier.code_dim, device="cuda", requires_grad=True
            )
            zero_prediction = carrier.readout(auxiliary, code=zero)
            zero_parity = _prediction_difference(zero_prediction, base)
            current_points = patch_center_points(
                zero_prediction["pts3d_in_other_view"], patch_size
            )
            online = symmetric_point_consistency(current_points, previous_points)
            metric = _metric(zero_prediction, gt_views[-1], config)
            online_gradient = torch.autograd.grad(
                online, zero, create_graph=False, retain_graph=True
            )[0]
            metric_gradient = torch.autograd.grad(
                metric, zero, create_graph=False
            )[0]
            token_feature, geometry_feature = _features(
                tokens, current_points.detach(), previous_points
            )
            labels = (online_gradient * metric_gradient > 0)
            records.append(
                {
                    "scene": scene,
                    "sequence": sequence,
                    "target_frame": int(target_frame),
                    "zero_code_max_abs_difference": zero_parity,
                    "token": token_feature[0].detach().cpu().numpy().astype(np.float32),
                    "geometry": geometry_feature[0].detach().cpu().numpy().astype(np.float32),
                    "labels": labels[0].detach().cpu().numpy(),
                }
            )
            print(
                json.dumps(
                    {
                        "collected": len(records),
                        "total": 16,
                        "scene": scene,
                        "sequence": sequence,
                        "target_frame": int(target_frame),
                        "positive_label_fraction": float(labels.float().mean()),
                    }
                ),
                flush=True,
            )
            del dataset, gt_views, base, auxiliary, previous_points, tokens, zero
            del zero_prediction, current_points, online, metric, online_gradient
            del metric_gradient, token_feature, geometry_feature, labels
            gc.collect()
            torch.cuda.empty_cache()

    scenes = sorted({record["scene"] for record in records})
    fold_models = {}
    label_rows = []
    for scene in scenes:
        train_records = [record for record in records if record["scene"] != scene]
        held_records = [record for record in records if record["scene"] == scene]
        fold_models[scene] = {
            name: _fit_moment(train_records, name) for name in FEATURE_SETS
        }
        for record_index, record in enumerate(held_records):
            token_score = _score(fold_models[scene]["token"], record["token"])
            geometry_score = _score(
                fold_models[scene]["geometry"], record["geometry"]
            )
            combined_feature = _feature(record, "combined")
            combined_score = _score(
                fold_models[scene]["combined"], combined_feature
            )
            generator = np.random.default_rng(
                seed + scenes.index(scene) * 100 + record_index
            )
            permutation = generator.permutation(len(record["geometry"]))
            shuffled_feature = np.concatenate(
                (record["token"], record["geometry"][permutation]), axis=-1
            )
            shuffled_score = _score(
                fold_models[scene]["combined"], shuffled_feature
            )
            label_rows.append(
                {
                    "scene": scene,
                    "sequence": record["sequence"],
                    "target_frame": record["target_frame"],
                    "token_balanced_accuracy": _balanced_accuracy(
                        token_score, record["labels"]
                    ),
                    "geometry_balanced_accuracy": _balanced_accuracy(
                        geometry_score, record["labels"]
                    ),
                    "combined_balanced_accuracy": _balanced_accuracy(
                        combined_score, record["labels"]
                    ),
                    "shuffled_geometry_balanced_accuracy": _balanced_accuracy(
                        shuffled_score, record["labels"]
                    ),
                }
            )

    record_map = {
        (record["sequence"], record["target_frame"]): record for record in records
    }
    label_map = {
        (row["sequence"], row["target_frame"]): row for row in label_rows
    }
    rows = []
    # Phase 2: apply OOF scales to the actual unchanged one-step readout.
    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        held_records = [record for record in records if record["scene"] == scene]
        held_order = {
            (record["sequence"], record["target_frame"]): index
            for index, record in enumerate(held_records)
        }
        for target_frame in config["data"]["target_frames"]:
            target_frame = int(target_frame)
            key = (sequence, target_frame)
            record = record_map[key]
            dataset, gt_views = _dataset_item(SevenScenes, config, sequence, target_frame)
            base, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
            tokens = auxiliary["decoder_patch_tokens"].float()
            zero = torch.zeros(
                1, tokens.shape[1], carrier.code_dim, device="cuda", requires_grad=True
            )
            zero_prediction = carrier.readout(auxiliary, code=zero)
            current_points = patch_center_points(
                zero_prediction["pts3d_in_other_view"], patch_size
            )
            online_before = symmetric_point_consistency(current_points, previous_points)
            metric_before = _metric(zero_prediction, gt_views[-1], config)
            online_gradient = torch.autograd.grad(
                online_before, zero, create_graph=False
            )[0].detach()
            token_feature, geometry_feature = _features(
                tokens, current_points.detach(), previous_points
            )
            token_np = token_feature[0].detach().cpu().numpy().astype(np.float32)
            geometry_np = geometry_feature[0].detach().cpu().numpy().astype(np.float32)
            models = fold_models[scene]
            token_score = _score(models["token"], token_np)
            geometry_score = _score(models["geometry"], geometry_np)
            combined_score = _score(
                models["combined"], np.concatenate((token_np, geometry_np), axis=-1)
            )
            generator = np.random.default_rng(
                seed + scenes.index(scene) * 100 + held_order[key]
            )
            permutation = generator.permutation(len(geometry_np))
            shuffled_score = _score(
                models["combined"],
                np.concatenate((token_np, geometry_np[permutation]), axis=-1),
            )
            scales = {
                "global": torch.ones_like(online_gradient),
                "token": torch.as_tensor(
                    1.0 + np.tanh(token_score), device="cuda", dtype=online_gradient.dtype
                )[None],
                "geometry": torch.as_tensor(
                    1.0 + np.tanh(geometry_score), device="cuda", dtype=online_gradient.dtype
                )[None],
                "combined": torch.as_tensor(
                    1.0 + np.tanh(combined_score), device="cuda", dtype=online_gradient.dtype
                )[None],
                "shuffled_geometry": torch.as_tensor(
                    1.0 + np.tanh(shuffled_score), device="cuda", dtype=online_gradient.dtype
                )[None],
                "oracle": torch.as_tensor(
                    record["labels"], device="cuda", dtype=online_gradient.dtype
                )[None],
            }
            policy_results = {}
            for policy in POLICIES:
                _, policy_results[policy] = _policy_result(
                    carrier,
                    auxiliary,
                    zero.detach(),
                    online_gradient,
                    scales[policy],
                    previous_points,
                    gt_views[-1],
                    step_size=step_size,
                    patch_size=patch_size,
                    minimum_depth=float(config["metric"]["minimum_depth_m"]),
                    maximum_depth=float(config["metric"]["maximum_depth_m"]),
                    online_before=online_before,
                    metric_before=metric_before,
                )
            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": target_frame,
                "zero_code_max_abs_difference": _prediction_difference(
                    zero_prediction, base
                ),
                **label_map[key],
                "policies": policy_results,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "evaluated": len(rows),
                        "total": 16,
                        "scene": scene,
                        "sequence": sequence,
                        "target_frame": target_frame,
                        "combined_metric_gain": policy_results["combined"]["metric_gain"],
                    }
                ),
                flush=True,
            )
            del dataset, gt_views, base, auxiliary, previous_points, tokens, zero
            del zero_prediction, current_points, online_before, metric_before
            del online_gradient, token_feature, geometry_feature, scales
            gc.collect()
            torch.cuda.empty_cache()

    scene_means = {
        policy: {
            metric: _scene_means(
                [
                    {"scene": row["scene"], metric: row["policies"][policy][metric]}
                    for row in rows
                ],
                metric,
            )
            for metric in ("online_loss_gain", "metric_gain")
        }
        for policy in POLICIES
    }
    means = {
        policy: {
            metric: float(np.mean(list(scene_means[policy][metric].values())))
            for metric in scene_means[policy]
        }
        for policy in POLICIES
    }
    accuracy_keys = (
        "token_balanced_accuracy",
        "geometry_balanced_accuracy",
        "combined_balanced_accuracy",
        "shuffled_geometry_balanced_accuracy",
    )
    accuracy_scene_means = {key: _scene_means(rows, key) for key in accuracy_keys}
    accuracy_means = {
        key: float(np.mean(list(value.values())))
        for key, value in accuracy_scene_means.items()
    }
    comparisons = {}
    for offset, control in enumerate(("global", "token", "shuffled_geometry")):
        values = [
            row["policies"]["combined"]["metric_gain"]
            - row["policies"][control]["metric_gain"]
            for row in rows
        ]
        comparisons[f"combined_minus_{control}"] = _bootstrap_mean(
            values,
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]) + offset,
        )
    combined_harm = float(
        np.mean([row["policies"]["combined"]["metric_gain"] < 0 for row in rows])
    )
    checks = {
        "exact_coverage": len(rows) == int(config["success"]["exact_anchors"])
        and len(scenes) == int(config["success"]["exact_scenes"])
        and len(fold_models) == int(config["success"]["exact_folds"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["zero_code_max_abs_difference"],
                *(row[key] for key in accuracy_keys),
                *(
                    item
                    for policy in POLICIES
                    for item in row["policies"][policy].values()
                ),
            )
        ),
        "zero_code_parity": max(row["zero_code_max_abs_difference"] for row in rows)
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "combined_accuracy_beats_token": accuracy_means["combined_balanced_accuracy"]
        > accuracy_means["token_balanced_accuracy"],
        "combined_accuracy_beats_shuffle": accuracy_means["combined_balanced_accuracy"]
        > accuracy_means["shuffled_geometry_balanced_accuracy"],
        "combined_online_descent_all_scenes": all(
            value > 0 for value in scene_means["combined"]["online_loss_gain"].values()
        ),
        "combined_metric_gain_all_scenes": all(
            value > 0 for value in scene_means["combined"]["metric_gain"].values()
        ),
        "combined_beats_global_all_scenes": all(
            scene_means["combined"]["metric_gain"][scene]
            > scene_means["global"]["metric_gain"][scene]
            for scene in scenes
        ),
        "combined_beats_token_all_scenes": all(
            scene_means["combined"]["metric_gain"][scene]
            > scene_means["token"]["metric_gain"][scene]
            for scene in scenes
        ),
        "combined_beats_shuffle_all_scenes": all(
            scene_means["combined"]["metric_gain"][scene]
            > scene_means["shuffled_geometry"]["metric_gain"][scene]
            for scene in scenes
        ),
        "combined_vs_global_positive_ci": comparisons["combined_minus_global"]["ci95"][0]
        > 0,
        "combined_vs_token_positive_ci": comparisons["combined_minus_token"]["ci95"][0]
        > 0,
        "combined_vs_shuffle_positive_ci": comparisons[
            "combined_minus_shuffled_geometry"
        ]["ci95"][0]
        > 0,
        "combined_harm_within_bound": combined_harm
        <= float(config["success"]["maximum_combined_harm_fraction"]),
        "no_method_checkpoint": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-056",
        "stage": "train_only_scene_oof_pairwise_geometry_observability",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "means": means,
        "scene_means": scene_means,
        "accuracy_means": accuracy_means,
        "accuracy_scene_means": accuracy_scene_means,
        "paired_comparisons": comparisons,
        "combined_harm_fraction": combined_harm,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "method_checkpoint_created": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
