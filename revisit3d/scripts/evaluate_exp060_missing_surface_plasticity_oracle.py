#!/usr/bin/env python3
"""Test oracle-paired local-code reuse under controlled missing evidence."""
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
    transport_code_3d,
)
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _model_view,
    _scene_means,
)
from revisit3d.scripts.evaluate_exp054_conditional_tangent_oracle import (
    _bootstrap_mean,
)
from revisit3d.scripts.evaluate_exp057_explicit_missing_surface_oracle import (
    _align_prediction,
    _depth_to_points,
    _erase_view,
    _forward_warp_points,
    _next_code,
    _points_numpy,
    _relative_error,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _online_loss(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    code: torch.Tensor,
    previous_points: torch.Tensor,
    patch_size: int,
) -> float:
    with torch.no_grad():
        prediction = carrier.readout(auxiliary, code=code)
        points = patch_center_points(
            prediction["pts3d_in_other_view"], patch_size
        )
        return float(symmetric_point_consistency(points, previous_points))


def _masked_patch_code(code: torch.Tensor, mask: np.ndarray, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    offset = patch_size // 2
    token_mask = torch.as_tensor(
        mask[offset::patch_size, offset::patch_size].reshape(1, -1, 1),
        device=code.device,
        dtype=code.dtype,
    )
    if token_mask.shape[1] != code.shape[1]:
        raise RuntimeError("erasure/token layout mismatch")
    return code * token_mask, token_mask.bool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-060_missing_surface_plasticity_oracle_v10.yaml",
    )
    parser.add_argument("--confirm-train-rgbd-erasure", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_erasure or not torch.cuda.is_available():
        raise SystemExit("EXP-060 requires train-RGB-D erasure confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    depth_path = Path(config["data"]["depth_preparation"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    exp058_path = Path(config["controls"]["exp058_result"])
    if output.exists():
        raise RuntimeError("EXP-060 result already exists")

    manifest = json.loads(manifest_path.read_text())
    depth_preparation = json.loads(depth_path.read_text())
    exp058 = json.loads(exp058_path.read_text())
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(exp058_path) == config["controls"]["exp058_result_sha256"]
        and exp058["experiment"] == "EXP-058"
        and exp058["validation_accessed"] is False
        and exp058["terminal_accessed"] is False
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and depth_preparation["validation_accessed"] is False
        and depth_preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and int(config["plasticity"]["current_steps"]) == 2
    ):
        raise RuntimeError("EXP-060 source-safe contract failed")

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
    reference = {
        (row["sequence"], int(row["target_frame"])): row
        for row in exp058["rows"]
    }
    rows = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame_value in config["data"]["target_frames"]:
            target_frame = int(target_frame_value)
            indices = list(
                range(target_frame - context_frames + 1, target_frame + 1)
            )
            spec = sequence + " " + " ".join(
                f"{index:06d}" for index in indices
            )
            dataset = SevenScenes(
                split="train",
                ROOT=config["data"]["root"],
                resolution=tuple(config["carrier"]["resolution"]),
                tuple_list=[spec],
                seed=seed,
            )
            gt_views = dataset[0]
            state = None
            source_previous_points = None
            source_prediction = None
            source_auxiliary = None
            for index in range(len(gt_views) - 1):
                with torch.no_grad():
                    prediction, state, auxiliary = carrier.step(
                        _model_view(gt_views[index], index), state
                    )
                if index == len(gt_views) - 3:
                    source_previous_points = patch_center_points(
                        prediction["pts3d_in_other_view"], patch_size
                    ).detach()
                if index == len(gt_views) - 2:
                    source_prediction = prediction
                    source_auxiliary = auxiliary
            if (
                source_previous_points is None
                or source_prediction is None
                or source_auxiliary is None
                or state is None
            ):
                raise RuntimeError("source adaptation context is incomplete")

            source_zero = torch.zeros(
                1,
                source_auxiliary["decoder_patch_tokens"].shape[1],
                carrier.code_dim,
                device="cuda",
            )
            source_code = _next_code(
                carrier,
                source_auxiliary,
                source_zero,
                source_previous_points,
                step_size=step_size,
                patch_size=patch_size,
            )
            source_online_base = _online_loss(
                carrier,
                source_auxiliary,
                source_zero,
                source_previous_points,
                patch_size,
            )
            source_online_adapted = _online_loss(
                carrier,
                source_auxiliary,
                source_code,
                source_previous_points,
                patch_size,
            )

            target_previous_points = patch_center_points(
                source_prediction["pts3d_in_other_view"], patch_size
            ).detach()
            erased_view, erased_mask = _erase_view(
                gt_views[-1], len(gt_views) - 1, config
            )
            with torch.no_grad():
                erased_prediction, _, target_auxiliary = carrier.step(
                    erased_view, state
                )
            target_zero = torch.zeros(
                1,
                target_auxiliary["decoder_patch_tokens"].shape[1],
                carrier.code_dim,
                device="cuda",
            )
            current_one = _next_code(
                carrier,
                target_auxiliary,
                target_zero,
                target_previous_points,
                step_size=step_size,
                patch_size=patch_size,
            )
            current_two = _next_code(
                carrier,
                target_auxiliary,
                current_one,
                target_previous_points,
                step_size=step_size,
                patch_size=patch_size,
            )

            source_base_points = patch_center_points(
                source_prediction["pts3d_in_other_view"], patch_size
            )
            target_base_points = patch_center_points(
                erased_prediction["pts3d_in_other_view"], patch_size
            )
            transported, transport_distance = transport_code_3d(
                source_base_points, source_code, target_base_points
            )
            transported_masked, token_mask = _masked_patch_code(
                transported, erased_mask, patch_size
            )
            untransported_masked, _ = _masked_patch_code(
                source_code, erased_mask, patch_size
            )
            masked_indices = torch.flatnonzero(token_mask[0, :, 0])
            generator = torch.Generator(device="cpu").manual_seed(
                int(config["controls"]["spatial_shuffle_seed"]) + len(rows)
            )
            permutation = torch.randperm(
                masked_indices.numel(), generator=generator
            ).to(masked_indices.device)
            shuffled_masked = torch.zeros_like(transported_masked)
            shuffled_masked[:, masked_indices] = transported_masked[
                :, masked_indices[permutation]
            ]

            candidate_codes = {
                "current_one": current_one,
                "current_two": current_two,
                "transported_memory": current_one + transported_masked,
                "untransported_memory": current_one + untransported_masked,
                "shuffled_memory": current_one + shuffled_masked,
            }
            with torch.no_grad():
                predictions = {
                    name: carrier.readout(target_auxiliary, code=code)
                    for name, code in candidate_codes.items()
                }

            online_losses = {
                "source_base": source_online_base,
                "source_adapted": source_online_adapted,
                "target_base": _online_loss(
                    carrier,
                    target_auxiliary,
                    target_zero,
                    target_previous_points,
                    patch_size,
                ),
                "target_current_one": _online_loss(
                    carrier,
                    target_auxiliary,
                    current_one,
                    target_previous_points,
                    patch_size,
                ),
                "target_current_two": _online_loss(
                    carrier,
                    target_auxiliary,
                    current_two,
                    target_previous_points,
                    patch_size,
                ),
            }

            target_depth = np.asarray(gt_views[-1]["depthmap"], dtype=np.float32)
            target_intrinsics = np.asarray(
                gt_views[-1]["camera_intrinsics"], dtype=np.float32
            )
            target_gt_pose = np.asarray(
                gt_views[-1]["camera_pose"], dtype=np.float32
            )
            source_depth = np.asarray(
                gt_views[-2]["depthmap"], dtype=np.float32
            )
            source_intrinsics = np.asarray(
                gt_views[-2]["camera_intrinsics"], dtype=np.float32
            )
            source_gt_pose = np.asarray(
                gt_views[-2]["camera_pose"], dtype=np.float32
            )
            height, width = target_depth.shape
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
                source_gt_pose,
                target_gt_pose,
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
            if supported_pixels < int(
                config["visibility"]["minimum_supported_pixels"]
            ):
                raise RuntimeError("EXP-060 insufficient evaluation support")

            aligned_erased, erased_scale = _align_prediction(
                _points_numpy(erased_prediction), target_depth, alignment_mask
            )
            errors = {}
            error_maps = {}
            erased_error, erased_map = _relative_error(
                aligned_erased, target_points, target_depth, evaluation_mask
            )
            errors["erased"] = erased_error
            error_maps["erased"] = erased_map
            alignment_scales = {"erased": erased_scale}
            for name, prediction in predictions.items():
                aligned, scale = _align_prediction(
                    _points_numpy(prediction), target_depth, alignment_mask
                )
                errors[name], error_maps[name] = _relative_error(
                    aligned, target_points, target_depth, evaluation_mask
                )
                alignment_scales[name] = scale
            best_map = np.minimum(
                error_maps["current_two"], error_maps["transported_memory"]
            )
            errors["best_memory"] = float(np.mean(best_map[evaluation_mask]))

            reference_row = reference[(sequence, target_frame)]
            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": target_frame,
                "supported_pixels": supported_pixels,
                "supported_pixels_match_exp058": (
                    supported_pixels == reference_row["supported_pixels"]
                ),
                "erased_reproduction_abs": abs(
                    errors["erased"] - reference_row["errors"]["erased"]
                ),
                "online_losses": online_losses,
                "errors": errors,
                "alignment_scales": alignment_scales,
                "mean_transport_distance": float(transport_distance.mean()),
                "masked_token_count": int(masked_indices.numel()),
                "memory_gain_vs_second": (
                    errors["current_two"] - errors["transported_memory"]
                ),
                "memory_gain_vs_untransported": (
                    errors["untransported_memory"]
                    - errors["transported_memory"]
                ),
                "memory_gain_vs_shuffle": (
                    errors["shuffled_memory"] - errors["transported_memory"]
                ),
                "best_memory_gain_vs_second": (
                    errors["current_two"] - errors["best_memory"]
                ),
                "exp058_surface_gain_vs_second": reference_row[
                    "predicted_only_gain_vs_second"
                ],
            }
            rows.append(row)
            print(
                json.dumps({"evaluated": len(rows), "total": 16, **row}),
                flush=True,
            )

            del dataset, gt_views, state, source_previous_points
            del source_prediction, source_auxiliary, source_zero, source_code
            del target_previous_points, erased_prediction, target_auxiliary
            del target_zero, current_one, current_two, transported
            del transported_masked, untransported_masked, shuffled_masked
            del predictions, candidate_codes
            gc.collect()
            torch.cuda.empty_cache()

    gain_keys = (
        "memory_gain_vs_second",
        "memory_gain_vs_untransported",
        "memory_gain_vs_shuffle",
        "best_memory_gain_vs_second",
    )
    scene_means = {key: _scene_means(rows, key) for key in gain_keys}
    means = {
        key: float(np.mean(list(scene_values.values())))
        for key, scene_values in scene_means.items()
    }
    intervals = {
        key: _bootstrap_mean(
            [row[key] for row in rows],
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]) + index,
        )
        for index, key in enumerate(gain_keys[:3])
    }
    harm = float(np.mean([row["memory_gain_vs_second"] < 0 for row in rows]))
    checks = {
        "exact_coverage": (
            len(rows) == int(config["success"]["exact_anchors"])
            and len({row["scene"] for row in rows})
            == int(config["success"]["exact_scenes"])
        ),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                *row["errors"].values(),
                *row["online_losses"].values(),
                *(row[key] for key in gain_keys),
                row["erased_reproduction_abs"],
                row["mean_transport_distance"],
            )
        ),
        "evaluation_support_matches_exp058": all(
            row["supported_pixels_match_exp058"] for row in rows
        ),
        "erased_base_reproduces_exp058": max(
            row["erased_reproduction_abs"] for row in rows
        )
        <= float(config["success"]["maximum_erased_reproduction_abs_difference"]),
        "source_online_step_descends_every_anchor": all(
            row["online_losses"]["source_adapted"]
            < row["online_losses"]["source_base"]
            for row in rows
        ),
        "target_first_online_step_descends_every_anchor": all(
            row["online_losses"]["target_current_one"]
            < row["online_losses"]["target_base"]
            for row in rows
        ),
        "target_second_online_step_descends_every_anchor": all(
            row["online_losses"]["target_current_two"]
            < row["online_losses"]["target_current_one"]
            for row in rows
        ),
        "memory_beats_second_all_scenes": all(
            value > 0 for value in scene_means["memory_gain_vs_second"].values()
        ),
        "memory_beats_second_positive_ci": intervals[
            "memory_gain_vs_second"
        ]["ci95"][0]
        > 0,
        "memory_beats_untransported_all_scenes": all(
            value > 0
            for value in scene_means["memory_gain_vs_untransported"].values()
        ),
        "memory_beats_untransported_positive_ci": intervals[
            "memory_gain_vs_untransported"
        ]["ci95"][0]
        > 0,
        "memory_beats_shuffle_all_scenes": all(
            value > 0 for value in scene_means["memory_gain_vs_shuffle"].values()
        ),
        "memory_beats_shuffle_positive_ci": intervals[
            "memory_gain_vs_shuffle"
        ]["ci95"][0]
        > 0,
        "memory_harm_within_bound": harm
        <= float(config["success"]["maximum_memory_harm_fraction"]),
        "no_parameter_fit": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-060",
        "stage": "train_only_no_fit_missing_surface_plasticity_oracle",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "means": means,
        "scene_means": scene_means,
        "intervals": intervals,
        "memory_harm_fraction": harm,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "parameter_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
