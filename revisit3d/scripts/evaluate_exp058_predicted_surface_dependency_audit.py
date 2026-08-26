#!/usr/bin/env python3
"""Remove GT pose/scale/visibility dependencies from EXP-057 fusion."""
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

from revisit3d.backbones import FrozenCUT3RCarrier, patch_center_points
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-058_predicted_surface_dependency_audit_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-audit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_audit or not torch.cuda.is_available():
        raise SystemExit("EXP-058 requires train-RGB-D audit confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    depth_path = Path(config["data"]["depth_preparation"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    exp057_path = Path(config["controls"]["exp057_result"])
    if output.exists():
        raise RuntimeError("EXP-058 result already exists")
    manifest = json.loads(manifest_path.read_text())
    depth_preparation = json.loads(depth_path.read_text())
    exp057 = json.loads(exp057_path.read_text())
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(exp057_path) == config["controls"]["exp057_result_sha256"]
        and exp057["experiment"] == "EXP-057"
        and exp057["registered_gate"]["passed"] is True
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and depth_preparation["validation_accessed"] is False
        and depth_preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-058 source-safe contract failed")

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
    # Import after the carrier has initialized dust3r's head registry. Importing
    # camera utilities first triggers the external repository's camera/head
    # circular import before either module has finished initialization.
    from dust3r.utils.camera import pose_encoding_to_camera
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    context_frames = int(config["data"]["context_frames"])
    reference = {
        (row["sequence"], int(row["target_frame"])): row for row in exp057["rows"]
    }
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
            erased_view, erased_mask = _erase_view(
                gt_views[-1], len(gt_views) - 1, config
            )
            with torch.no_grad():
                erased_prediction, _, auxiliary = carrier.step(erased_view, state)

            zero = torch.zeros(
                1,
                auxiliary["decoder_patch_tokens"].shape[1],
                carrier.code_dim,
                device="cuda",
            )
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
                source_pose = pose_encoding_to_camera(
                    source_prediction["camera_pose"].float()
                )[0].cpu().numpy()
                target_pose = pose_encoding_to_camera(
                    erased_prediction["camera_pose"].float()
                )[0].cpu().numpy()

            target_depth = np.asarray(gt_views[-1]["depthmap"], dtype=np.float32)
            target_intrinsics = np.asarray(
                gt_views[-1]["camera_intrinsics"], dtype=np.float32
            )
            target_gt_pose = np.asarray(gt_views[-1]["camera_pose"], dtype=np.float32)
            source_depth = np.asarray(gt_views[-2]["depthmap"], dtype=np.float32)
            source_intrinsics = np.asarray(
                gt_views[-2]["camera_intrinsics"], dtype=np.float32
            )
            source_gt_pose = np.asarray(gt_views[-2]["camera_pose"], dtype=np.float32)
            height, width = target_depth.shape
            target_valid = (
                np.isfinite(target_depth)
                & (target_depth >= minimum_depth)
                & (target_depth <= maximum_depth)
            )
            alignment_mask = target_valid & ~erased_mask
            target_points = _depth_to_points(target_depth, target_intrinsics)

            # GT is used only to reproduce the immutable evaluation support.
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
            if supported_pixels < int(config["visibility"]["minimum_supported_pixels"]):
                raise RuntimeError("EXP-058 insufficient evaluation support")

            source_points = _points_numpy(source_prediction)
            warped_predicted_depth = _forward_warp_points(
                source_points,
                source_pose,
                target_pose,
                target_intrinsics,
                (height, width),
                minimum_depth=1e-6,
                maximum_depth=1e6,
            )
            policy_support = erased_mask & np.isfinite(warped_predicted_depth)
            evaluation_coverage = float(
                (policy_support & evaluation_mask).sum() / supported_pixels
            )

            raw_two = _points_numpy(two_prediction)
            predicted_past_points = _depth_to_points(
                warped_predicted_depth, target_intrinsics
            )
            fused_raw = raw_two.copy()
            fused_raw[policy_support] = predicted_past_points[policy_support]
            fused, fused_scale = _align_prediction(
                fused_raw, target_depth, alignment_mask
            )

            policy_indices = np.flatnonzero(policy_support.reshape(-1))
            shuffled_depth = warped_predicted_depth.copy().reshape(-1)
            generator = np.random.default_rng(seed + len(rows))
            shuffled_depth[policy_indices] = shuffled_depth[
                generator.permutation(policy_indices)
            ]
            shuffled_depth = shuffled_depth.reshape(height, width)
            shuffled_points = _depth_to_points(shuffled_depth, target_intrinsics)
            shuffled_raw = raw_two.copy()
            shuffled_raw[policy_support] = shuffled_points[policy_support]
            shuffled, shuffled_scale = _align_prediction(
                shuffled_raw, target_depth, alignment_mask
            )

            aligned_erased, erased_scale = _align_prediction(
                _points_numpy(erased_prediction), target_depth, alignment_mask
            )
            aligned_one, one_scale = _align_prediction(
                _points_numpy(one_prediction), target_depth, alignment_mask
            )
            aligned_two, two_scale = _align_prediction(
                raw_two, target_depth, alignment_mask
            )
            errors = {}
            error_maps = {}
            for policy, points in (
                ("erased", aligned_erased),
                ("current_one", aligned_one),
                ("current_two", aligned_two),
                ("predicted_only_fusion", fused),
                ("shuffled_predicted_only_fusion", shuffled),
            ):
                errors[policy], error_maps[policy] = _relative_error(
                    points, target_points, target_depth, evaluation_mask
                )
            best_map = np.minimum(
                error_maps["current_two"], error_maps["predicted_only_fusion"]
            )
            errors["best_predicted_only_fusion"] = float(
                np.mean(best_map[evaluation_mask])
            )

            exp057_row = reference[(sequence, target_frame)]
            reference_second = float(exp057_row["errors"]["current_two"])
            oracle_gain = float(exp057_row["predicted_past_gain_vs_second"])
            predicted_gain = errors["current_two"] - errors["predicted_only_fusion"]
            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": target_frame,
                "supported_pixels": supported_pixels,
                "predicted_only_coverage": evaluation_coverage,
                "errors": errors,
                "alignment_scales": {
                    "erased": erased_scale,
                    "current_one": one_scale,
                    "current_two": two_scale,
                    "predicted_only_fusion": fused_scale,
                    "shuffled_predicted_only_fusion": shuffled_scale,
                },
                "exp057_second_reproduction_abs": abs(
                    errors["current_two"] - reference_second
                ),
                "exp057_oracle_predicted_gain": oracle_gain,
                "predicted_only_gain_vs_second": predicted_gain,
                "predicted_only_gain_vs_shuffle": errors[
                    "shuffled_predicted_only_fusion"
                ]
                - errors["predicted_only_fusion"],
                "best_predicted_only_gain_vs_second": errors["current_two"]
                - errors["best_predicted_only_fusion"],
                "oracle_gain_retention": predicted_gain / max(oracle_gain, 1e-12),
            }
            rows.append(row)
            print(json.dumps({"evaluated": len(rows), "total": 16, **row}), flush=True)

            del dataset, gt_views, state, source_prediction, previous_points
            del erased_prediction, auxiliary, zero, first_code, second_code
            del one_prediction, two_prediction
            gc.collect()
            torch.cuda.empty_cache()

    keys = (
        "predicted_only_coverage",
        "predicted_only_gain_vs_second",
        "predicted_only_gain_vs_shuffle",
        "best_predicted_only_gain_vs_second",
        "oracle_gain_retention",
    )
    scene_means = {key: _scene_means(rows, key) for key in keys}
    means = {key: float(np.mean(list(value.values()))) for key, value in scene_means.items()}
    intervals = {
        key: _bootstrap_mean(
            [row[key] for row in rows],
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]) + index,
        )
        for index, key in enumerate(
            ("predicted_only_gain_vs_second", "predicted_only_gain_vs_shuffle")
        )
    }
    harm = float(np.mean([row["predicted_only_gain_vs_second"] < 0 for row in rows]))
    checks = {
        "exact_coverage": len(rows) == int(config["success"]["exact_anchors"])
        and len({row["scene"] for row in rows})
        == int(config["success"]["exact_scenes"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["predicted_only_coverage"],
                row["exp057_second_reproduction_abs"],
                *row["errors"].values(),
                *(row[key] for key in keys[1:]),
            )
        ),
        "exp057_second_reproduction": max(
            row["exp057_second_reproduction_abs"] for row in rows
        )
        <= float(config["success"]["maximum_exp057_reproduction_abs_difference"]),
        "minimum_predicted_only_coverage": min(
            row["predicted_only_coverage"] for row in rows
        )
        >= float(config["visibility"]["minimum_predicted_coverage"]),
        "predicted_only_beats_second_all_scenes": all(
            value > 0
            for value in scene_means["predicted_only_gain_vs_second"].values()
        ),
        "predicted_only_beats_second_positive_ci": intervals[
            "predicted_only_gain_vs_second"
        ]["ci95"][0]
        > 0,
        "predicted_only_beats_shuffle_all_scenes": all(
            value > 0
            for value in scene_means["predicted_only_gain_vs_shuffle"].values()
        ),
        "predicted_only_beats_shuffle_positive_ci": intervals[
            "predicted_only_gain_vs_shuffle"
        ]["ci95"][0]
        > 0,
        "predicted_only_harm_within_bound": harm
        <= float(config["success"]["maximum_predicted_fusion_harm_fraction"]),
        "minimum_oracle_gain_retention": means["oracle_gain_retention"]
        >= float(config["success"]["minimum_oracle_gain_retention"]),
        "no_parameter_fit": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-058",
        "stage": "train_only_no_fit_predicted_surface_dependency_audit",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "means": means,
        "scene_means": scene_means,
        "intervals": intervals,
        "predicted_only_harm_fraction": harm,
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
