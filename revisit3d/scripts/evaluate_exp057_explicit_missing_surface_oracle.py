#!/usr/bin/env python3
"""Controlled-erasure explicit past-surface premise for EXP-057."""
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

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
)
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _model_view,
    _rms_normalize,
    _scene_means,
)
from revisit3d.scripts.evaluate_exp054_conditional_tangent_oracle import (
    _bootstrap_mean,
    _prediction_difference,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


POLICIES = (
    "clean",
    "erased",
    "current_one",
    "current_two",
    "gt_past_fusion",
    "predicted_past_fusion",
    "shuffled_past_fusion",
    "best_predicted_past_fusion",
)


def _depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    points = np.stack(
        (
            (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ),
        axis=-1,
    )
    return points.astype(np.float32)


def _camera_to_camera(
    points: np.ndarray, source_pose: np.ndarray, target_pose: np.ndarray
) -> np.ndarray:
    flat = points.reshape(-1, 3)
    world = flat @ source_pose[:3, :3].T + source_pose[:3, 3]
    target = (world - target_pose[:3, 3]) @ target_pose[:3, :3]
    return target.reshape(points.shape).astype(np.float32)


def _forward_warp_points(
    source_points: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    target_intrinsics: np.ndarray,
    target_shape: tuple[int, int],
    *,
    minimum_depth: float,
    maximum_depth: float,
) -> np.ndarray:
    target_points = _camera_to_camera(source_points, source_pose, target_pose).reshape(-1, 3)
    z = target_points[:, 2]
    valid = (
        np.isfinite(target_points).all(axis=-1)
        & (z >= minimum_depth)
        & (z <= maximum_depth)
    )
    x = target_points[:, 0]
    y = target_points[:, 1]
    u = np.rint(target_intrinsics[0, 0] * x / np.maximum(z, 1e-6) + target_intrinsics[0, 2]).astype(np.int64)
    v = np.rint(target_intrinsics[1, 1] * y / np.maximum(z, 1e-6) + target_intrinsics[1, 2]).astype(np.int64)
    height, width = target_shape
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    linear = v[valid] * width + u[valid]
    buffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(buffer, linear, z[valid].astype(np.float32))
    return buffer.reshape(height, width)


def _central_mask(height: int, width: int, config: dict) -> np.ndarray:
    mask_height = int(round(height * float(config["erasure"]["height_fraction"])))
    mask_width = int(round(width * float(config["erasure"]["width_fraction"])))
    top = (height - mask_height) // 2
    left = (width - mask_width) // 2
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + mask_height, left : left + mask_width] = True
    return mask


def _erase_view(view: dict, index: int, config: dict) -> tuple[dict, np.ndarray]:
    model_view = _model_view(view, index)
    image = model_view["img"].clone()
    height, width = image.shape[-2:]
    mask = _central_mask(height, width, config)
    tensor_mask = torch.as_tensor(mask, device=image.device)
    image[..., tensor_mask] = float(config["erasure"]["normalized_fill"])
    model_view["img"] = image
    return model_view, mask


def _align_prediction(
    points: np.ndarray,
    target_depth: np.ndarray,
    alignment_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    predicted_depth = points[..., 2]
    valid = (
        alignment_mask
        & np.isfinite(points).all(axis=-1)
        & np.isfinite(target_depth)
        & (predicted_depth > 1e-6)
        & (target_depth > 0)
    )
    if int(valid.sum()) < 2048:
        raise RuntimeError("EXP-057 has insufficient visible alignment pixels")
    scale = float(np.median(target_depth[valid]) / np.median(predicted_depth[valid]))
    return points * scale, scale


def _relative_error(
    points: np.ndarray,
    target_points: np.ndarray,
    target_depth: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    error = np.linalg.norm(points - target_points, axis=-1) / np.maximum(target_depth, 1e-6)
    return float(np.mean(error[evaluation_mask])), error


def _next_code(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    code: torch.Tensor,
    previous_points: torch.Tensor,
    *,
    step_size: float,
    patch_size: int,
) -> torch.Tensor:
    current = code.detach().clone().requires_grad_(True)
    prediction = carrier.readout(auxiliary, code=current)
    loss = symmetric_point_consistency(
        patch_center_points(prediction["pts3d_in_other_view"], patch_size),
        previous_points,
    )
    gradient = torch.autograd.grad(loss, current, create_graph=False)[0]
    return current.detach() - step_size * _rms_normalize(gradient.detach())


def _points_numpy(prediction: dict) -> np.ndarray:
    return prediction["pts3d_in_self_view"][0].detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-057_explicit_missing_surface_oracle_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-erasure", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_erasure or not torch.cuda.is_available():
        raise SystemExit("EXP-057 requires train-RGB-D erasure confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    depth_path = Path(config["data"]["depth_preparation"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-057 result already exists")
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
        and int(config["plasticity"]["current_steps"]) == 2
    ):
        raise RuntimeError("EXP-057 source-safe contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
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
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    context_frames = int(config["data"]["context_frames"])
    rows = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame in config["data"]["target_frames"]:
            target_frame = int(target_frame)
            indices = list(range(target_frame - context_frames + 1, target_frame + 1))
            spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
            dataset = SevenScenes(
                split="train",
                ROOT=config["data"]["root"],
                resolution=tuple(config["carrier"]["resolution"]),
                tuple_list=[spec],
                seed=seed,
            )
            gt_views = dataset[0]
            state = None
            source_prediction = None
            for index in range(len(gt_views) - 1):
                with torch.no_grad():
                    prediction, state, _ = carrier.step(
                        _model_view(gt_views[index], index), state
                    )
                if index == len(gt_views) - 2:
                    source_prediction = prediction
            assert source_prediction is not None and state is not None
            previous_points = patch_center_points(
                source_prediction["pts3d_in_other_view"], patch_size
            ).detach()

            clean_view = _model_view(gt_views[-1], len(gt_views) - 1)
            erased_view, erased_mask = _erase_view(
                gt_views[-1], len(gt_views) - 1, config
            )
            with torch.no_grad():
                clean_prediction, _, _ = carrier.step(clean_view, state)
                erased_prediction, _, auxiliary = carrier.step(erased_view, state)

            zero = torch.zeros(
                1,
                auxiliary["decoder_patch_tokens"].shape[1],
                carrier.code_dim,
                device="cuda",
                requires_grad=True,
            )
            zero_prediction = carrier.readout(auxiliary, code=zero)
            zero_parity = _prediction_difference(zero_prediction, erased_prediction)
            first_code = _next_code(
                carrier,
                auxiliary,
                zero,
                previous_points,
                step_size=step_size,
                patch_size=patch_size,
            )
            second_code = _next_code(
                carrier,
                auxiliary,
                first_code,
                previous_points,
                step_size=step_size,
                patch_size=patch_size,
            )
            with torch.no_grad():
                one_prediction = carrier.readout(auxiliary, code=first_code)
                two_prediction = carrier.readout(auxiliary, code=second_code)

            target_depth = np.asarray(gt_views[-1]["depthmap"], dtype=np.float32)
            target_intrinsics = np.asarray(
                gt_views[-1]["camera_intrinsics"], dtype=np.float32
            )
            target_pose = np.asarray(gt_views[-1]["camera_pose"], dtype=np.float32)
            source_depth = np.asarray(gt_views[-2]["depthmap"], dtype=np.float32)
            source_intrinsics = np.asarray(
                gt_views[-2]["camera_intrinsics"], dtype=np.float32
            )
            source_pose = np.asarray(gt_views[-2]["camera_pose"], dtype=np.float32)
            height, width = target_depth.shape
            if erased_mask.shape != (height, width):
                raise RuntimeError("EXP-057 erased/model metric grids differ")
            target_valid = (
                np.isfinite(target_depth)
                & (target_depth >= minimum_depth)
                & (target_depth <= maximum_depth)
            )
            alignment_mask = target_valid & ~erased_mask
            target_points = _depth_to_points(target_depth, target_intrinsics)

            source_gt_points = _depth_to_points(source_depth, source_intrinsics)
            warped_gt_depth = _forward_warp_points(
                source_gt_points,
                source_pose,
                target_pose,
                target_intrinsics,
                (height, width),
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            evaluation_mask = (
                erased_mask
                & target_valid
                & np.isfinite(warped_gt_depth)
                & (
                    np.abs(warped_gt_depth - target_depth)
                    <= float(config["visibility"]["depth_consistency_m"])
                )
            )
            supported_pixels = int(evaluation_mask.sum())
            if supported_pixels < int(config["visibility"]["minimum_supported_pixels"]):
                raise RuntimeError(
                    f"EXP-057 insufficient supported pixels: {supported_pixels}"
                )

            source_predicted_points = _points_numpy(source_prediction)
            source_scale_valid = (
                np.isfinite(source_predicted_points).all(axis=-1)
                & (source_predicted_points[..., 2] > 1e-6)
                & (source_depth >= minimum_depth)
                & (source_depth <= maximum_depth)
            )
            source_scale = float(
                np.median(source_depth[source_scale_valid])
                / np.median(source_predicted_points[..., 2][source_scale_valid])
            )
            warped_predicted_depth = _forward_warp_points(
                source_predicted_points * source_scale,
                source_pose,
                target_pose,
                target_intrinsics,
                (height, width),
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            predicted_support = evaluation_mask & np.isfinite(warped_predicted_depth)
            predicted_coverage = float(predicted_support.sum() / supported_pixels)

            aligned = {}
            alignment_scales = {}
            for policy, prediction in (
                ("clean", clean_prediction),
                ("erased", erased_prediction),
                ("current_one", one_prediction),
                ("current_two", two_prediction),
            ):
                aligned[policy], alignment_scales[policy] = _align_prediction(
                    _points_numpy(prediction), target_depth, alignment_mask
                )

            current_two_points = aligned["current_two"]
            gt_past_points = _depth_to_points(warped_gt_depth, target_intrinsics)
            predicted_past_points = _depth_to_points(
                warped_predicted_depth, target_intrinsics
            )
            gt_fusion = current_two_points.copy()
            gt_fusion[evaluation_mask] = gt_past_points[evaluation_mask]
            predicted_fusion = current_two_points.copy()
            predicted_fusion[predicted_support] = predicted_past_points[predicted_support]

            support_indices = np.flatnonzero(predicted_support.reshape(-1))
            shuffled_depth = warped_predicted_depth.copy().reshape(-1)
            generator = np.random.default_rng(seed + len(rows))
            shuffled_depth[support_indices] = shuffled_depth[
                generator.permutation(support_indices)
            ]
            shuffled_depth = shuffled_depth.reshape(height, width)
            shuffled_points = _depth_to_points(shuffled_depth, target_intrinsics)
            shuffled_fusion = current_two_points.copy()
            shuffled_fusion[predicted_support] = shuffled_points[predicted_support]

            errors = {}
            error_maps = {}
            for policy in ("clean", "erased", "current_one", "current_two"):
                errors[policy], error_maps[policy] = _relative_error(
                    aligned[policy], target_points, target_depth, evaluation_mask
                )
            errors["gt_past_fusion"], error_maps["gt_past_fusion"] = _relative_error(
                gt_fusion, target_points, target_depth, evaluation_mask
            )
            errors["predicted_past_fusion"], error_maps[
                "predicted_past_fusion"
            ] = _relative_error(
                predicted_fusion, target_points, target_depth, evaluation_mask
            )
            errors["shuffled_past_fusion"], error_maps[
                "shuffled_past_fusion"
            ] = _relative_error(
                shuffled_fusion, target_points, target_depth, evaluation_mask
            )
            best_error_map = np.minimum(
                error_maps["current_two"], error_maps["predicted_past_fusion"]
            )
            errors["best_predicted_past_fusion"] = float(
                np.mean(best_error_map[evaluation_mask])
            )
            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": target_frame,
                "supported_pixels": supported_pixels,
                "predicted_past_coverage": predicted_coverage,
                "zero_code_max_abs_difference": zero_parity,
                "source_predicted_metric_scale": source_scale,
                "target_alignment_scales": alignment_scales,
                "errors": errors,
                "erasure_damage": errors["erased"] - errors["clean"],
                "gt_past_gain_vs_second": errors["current_two"]
                - errors["gt_past_fusion"],
                "predicted_past_gain_vs_second": errors["current_two"]
                - errors["predicted_past_fusion"],
                "predicted_past_gain_vs_shuffle": errors["shuffled_past_fusion"]
                - errors["predicted_past_fusion"],
                "best_predicted_past_gain_vs_second": errors["current_two"]
                - errors["best_predicted_past_fusion"],
            }
            rows.append(row)
            print(json.dumps({"evaluated": len(rows), "total": 16, **row}), flush=True)

            del dataset, gt_views, state, source_prediction, previous_points
            del clean_prediction, erased_prediction, auxiliary, zero, zero_prediction
            del first_code, second_code, one_prediction, two_prediction
            gc.collect()
            torch.cuda.empty_cache()

    keys = (
        "erasure_damage",
        "gt_past_gain_vs_second",
        "predicted_past_gain_vs_second",
        "predicted_past_gain_vs_shuffle",
        "best_predicted_past_gain_vs_second",
        "predicted_past_coverage",
    )
    scene_means = {key: _scene_means(rows, key) for key in keys}
    means = {key: float(np.mean(list(value.values()))) for key, value in scene_means.items()}
    error_scene_means = {
        policy: _scene_means(
            [{"scene": row["scene"], "value": row["errors"][policy]} for row in rows],
            "value",
        )
        for policy in POLICIES
    }
    intervals = {
        key: _bootstrap_mean(
            [row[key] for row in rows],
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]) + index,
        )
        for index, key in enumerate(
            (
                "erasure_damage",
                "gt_past_gain_vs_second",
                "predicted_past_gain_vs_second",
                "predicted_past_gain_vs_shuffle",
            )
        )
    }
    predicted_harm = float(
        np.mean([row["predicted_past_gain_vs_second"] < 0 for row in rows])
    )
    checks = {
        "exact_coverage": len(rows) == int(config["success"]["exact_anchors"])
        and len({row["scene"] for row in rows})
        == int(config["success"]["exact_scenes"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["predicted_past_coverage"],
                row["zero_code_max_abs_difference"],
                *row["errors"].values(),
                *(row[key] for key in keys[:-1]),
            )
        ),
        "minimum_supported_pixels": min(row["supported_pixels"] for row in rows)
        >= int(config["visibility"]["minimum_supported_pixels"]),
        "minimum_predicted_coverage": min(
            row["predicted_past_coverage"] for row in rows
        )
        >= float(config["visibility"]["minimum_predicted_coverage"]),
        "zero_code_parity": max(row["zero_code_max_abs_difference"] for row in rows)
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "erasure_worse_all_scenes": all(
            value > 0 for value in scene_means["erasure_damage"].values()
        ),
        "erasure_damage_positive_ci": intervals["erasure_damage"]["ci95"][0] > 0,
        "gt_past_beats_second_all_scenes": all(
            value > 0 for value in scene_means["gt_past_gain_vs_second"].values()
        ),
        "gt_past_beats_second_positive_ci": intervals[
            "gt_past_gain_vs_second"
        ]["ci95"][0]
        > 0,
        "predicted_past_beats_second_all_scenes": all(
            value > 0
            for value in scene_means["predicted_past_gain_vs_second"].values()
        ),
        "predicted_past_beats_second_positive_ci": intervals[
            "predicted_past_gain_vs_second"
        ]["ci95"][0]
        > 0,
        "predicted_past_beats_shuffle_all_scenes": all(
            value > 0
            for value in scene_means["predicted_past_gain_vs_shuffle"].values()
        ),
        "predicted_past_beats_shuffle_positive_ci": intervals[
            "predicted_past_gain_vs_shuffle"
        ]["ci95"][0]
        > 0,
        "predicted_past_harm_within_bound": predicted_harm
        <= float(config["success"]["maximum_predicted_fusion_harm_fraction"]),
        "no_parameter_fit": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-057",
        "stage": "train_only_controlled_erasure_explicit_surface_oracle",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "means": means,
        "scene_means": scene_means,
        "error_scene_means": error_scene_means,
        "intervals": intervals,
        "predicted_past_harm_fraction": predicted_harm,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "parameter_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
