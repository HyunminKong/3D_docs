#!/usr/bin/env python3
"""No-fit function-space plasticity transport premise for EXP-067."""
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
    _rms_normalize,
)
from revisit3d.scripts.evaluate_exp057_explicit_missing_surface_oracle import (
    _depth_to_points,
    _next_code,
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
        points = patch_center_points(prediction["pts3d_in_other_view"], patch_size)
        return float(symmetric_point_consistency(points, previous_points))


def _function_pullback(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    initial_code: torch.Tensor,
    desired_points: torch.Tensor,
    step_size: float,
    patch_size: int,
) -> tuple[torch.Tensor, float, float]:
    code = initial_code.detach().clone().requires_grad_(True)
    prediction = carrier.readout(auxiliary, code=code)
    points = patch_center_points(prediction["pts3d_in_other_view"], patch_size)
    loss = torch.linalg.vector_norm(points - desired_points, dim=-1).mean()
    gradient = torch.autograd.grad(loss, code, create_graph=False)[0]
    updated = code.detach() - step_size * _rms_normalize(gradient.detach())
    with torch.no_grad():
        updated_points = patch_center_points(
            carrier.readout(auxiliary, code=updated)["pts3d_in_other_view"],
            patch_size,
        )
        updated_loss = torch.linalg.vector_norm(
            updated_points - desired_points, dim=-1
        ).mean()
    return updated, float(loss), float(updated_loss)


def _metric_errors(
    predictions: dict[str, dict[str, torch.Tensor]],
    target_view: dict,
    minimum_depth: float,
    maximum_depth: float,
    minimum_valid: int,
) -> tuple[dict[str, float], dict[str, float], int]:
    depth = np.asarray(target_view["depthmap"], dtype=np.float64)
    target = _depth_to_points(
        depth.astype(np.float32),
        np.asarray(target_view["camera_intrinsics"], dtype=np.float32),
    ).astype(np.float64)
    arrays = {
        name: prediction["pts3d_in_self_view"][0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
        for name, prediction in predictions.items()
    }
    valid = (
        np.isfinite(target).all(axis=-1)
        & (depth >= minimum_depth)
        & (depth <= maximum_depth)
    )
    for points in arrays.values():
        valid &= np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-6)
    if int(valid.sum()) < minimum_valid:
        raise RuntimeError("EXP-067 has insufficient common target pixels")
    errors = {}
    scales = {}
    for name, points in arrays.items():
        scale = float(np.median(depth[valid]) / np.median(points[..., 2][valid]))
        relative = np.linalg.norm(scale * points[valid] - target[valid], axis=-1) / np.maximum(
            depth[valid], minimum_depth
        )
        errors[name] = float(relative.mean())
        scales[name] = scale
    return errors, scales, int(valid.sum())


def _scene_means(rows: list[dict], key: str) -> dict[str, float]:
    return {
        scene: float(np.mean([row[key] for row in rows if row["scene"] == scene]))
        for scene in sorted({row["scene"] for row in rows})
    }


def _stratified_interval(
    rows: list[dict], key: str, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in rows})
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        scene_values = []
        for scene in scenes:
            scene_rows = by_scene[scene]
            selected = rng.integers(0, len(scene_rows), size=len(scene_rows))
            scene_values.append(np.mean([scene_rows[index][key] for index in selected]))
        draws[sample] = np.mean(scene_values)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-067_function_space_plasticity_transport_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-function-transport", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_function_transport or not torch.cuda.is_available():
        raise SystemExit("EXP-067 requires train-RGB-D function-transport confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-067 result already exists")
    if not preparation_path.exists():
        raise RuntimeError("EXP-067 selected train depths have not been registered")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    pairs = list(config["data"]["pairs"])
    selected_frames = {
        (pair["sequence"], int(frame) + offset)
        for pair in pairs
        for frame in (pair["source"], pair["target"])
        for offset in (-1, 0)
    }
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and {pair["sequence"] for pair in pairs}.issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and len(pairs) == int(config["success"]["exact_pairs"])
        and all(int(pair["target"]) - int(pair["source"]) >= 50 for pair in pairs)
        and config["data"]["pair_selection"]["metadata_only"] is True
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and preparation["experiment"] == "EXP-067"
        and preparation["selected_frames"] == len(selected_frames)
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
    ):
        raise RuntimeError("EXP-067 source-safe contract failed")

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
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
        update_type=config["carrier"]["mode"],
    ).cuda()
    carrier.eval()
    carrier.model.requires_grad_(False)
    carrier.residual.requires_grad_(False)
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    rows = []
    torch.cuda.reset_peak_memory_stats()

    for pair_index, pair in enumerate(pairs):
        sequence = pair["sequence"]
        scene = sequence.split("/", 1)[0]
        source = int(pair["source"])
        target = int(pair["target"])
        indices = [source - 1, source, target - 1, target]
        tuple_spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
        dataset = SevenScenes(
            split="train",
            ROOT=config["data"]["root"],
            resolution=tuple(config["carrier"]["resolution"]),
            tuple_list=[tuple_spec],
            seed=seed,
        )
        views = dataset[0]
        state = None
        predictions = []
        auxiliaries = []
        with torch.no_grad():
            for index, view in enumerate(views):
                prediction, state, auxiliary = carrier.step(_model_view(view, index), state)
                predictions.append(prediction)
                auxiliaries.append(auxiliary)

        tokens = auxiliaries[1]["decoder_patch_tokens"].shape[1]
        source_zero = torch.zeros(1, tokens, carrier.code_dim, device="cuda")
        target_zero = torch.zeros_like(source_zero)
        source_previous = patch_center_points(
            predictions[0]["pts3d_in_other_view"], patch_size
        ).detach()
        source_base_points = patch_center_points(
            predictions[1]["pts3d_in_other_view"], patch_size
        ).detach()
        source_code = _next_code(
            carrier,
            auxiliaries[1],
            source_zero,
            source_previous,
            step_size=step_size,
            patch_size=patch_size,
        )
        with torch.no_grad():
            source_adapted_prediction = carrier.readout(auxiliaries[1], code=source_code)
            source_adapted_points = patch_center_points(
                source_adapted_prediction["pts3d_in_other_view"], patch_size
            )
        source_displacement = (source_adapted_points - source_base_points).detach()

        target_previous = patch_center_points(
            predictions[2]["pts3d_in_other_view"], patch_size
        ).detach()
        target_base_points = patch_center_points(
            predictions[3]["pts3d_in_other_view"], patch_size
        ).detach()
        current_one = _next_code(
            carrier,
            auxiliaries[3],
            target_zero,
            target_previous,
            step_size=step_size,
            patch_size=patch_size,
        )
        current_two = _next_code(
            carrier,
            auxiliaries[3],
            current_one,
            target_previous,
            step_size=step_size,
            patch_size=patch_size,
        )
        with torch.no_grad():
            current_one_prediction = carrier.readout(auxiliaries[3], code=current_one)
            current_one_points = patch_center_points(
                current_one_prediction["pts3d_in_other_view"], patch_size
            ).detach()

        transported_code, code_transport_distance = transport_code_3d(
            source_base_points, source_code, target_base_points
        )
        transported_displacement, displacement_transport_distance = transport_code_3d(
            source_base_points, source_displacement, target_base_points
        )
        untransported_displacement = source_displacement
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["controls"]["spatial_shuffle_seed"]) + pair_index
        )
        permutation = torch.randperm(tokens, generator=generator).to("cuda")
        shuffled_displacement = transported_displacement[:, permutation]

        function_code, function_before, function_after = _function_pullback(
            carrier,
            auxiliaries[3],
            current_one,
            current_one_points + transported_displacement,
            step_size,
            patch_size,
        )
        untransported_code, untransported_before, untransported_after = _function_pullback(
            carrier,
            auxiliaries[3],
            current_one,
            current_one_points + untransported_displacement,
            step_size,
            patch_size,
        )
        shuffled_code, shuffled_before, shuffled_after = _function_pullback(
            carrier,
            auxiliaries[3],
            current_one,
            current_one_points + shuffled_displacement,
            step_size,
            patch_size,
        )
        candidate_codes = {
            "current_one": current_one,
            "current_two": current_two,
            "direct_code_transport": current_one + transported_code,
            "function_transport": function_code,
            "untransported_function": untransported_code,
            "shuffled_function": shuffled_code,
        }
        with torch.no_grad():
            candidate_predictions = {
                name: carrier.readout(auxiliaries[3], code=code)
                for name, code in candidate_codes.items()
            }
            source_zero_prediction = carrier.readout(auxiliaries[1], code=source_zero)
            target_zero_prediction = carrier.readout(auxiliaries[3], code=target_zero)
        zero_parity = max(
            float(
                (
                    reference[key].detach().float() - zeroed[key].detach().float()
                ).abs().max()
            )
            for reference, zeroed in (
                (predictions[1], source_zero_prediction),
                (predictions[3], target_zero_prediction),
            )
            for key in ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")
            if key in reference and key in zeroed
        )
        errors, scales, valid_pixels = _metric_errors(
            candidate_predictions,
            views[3],
            float(config["metric"]["minimum_depth_m"]),
            float(config["metric"]["maximum_depth_m"]),
            int(config["metric"]["minimum_valid_pixels"]),
        )
        online = {
            "source_base": _online_loss(
                carrier, auxiliaries[1], source_zero, source_previous, patch_size
            ),
            "source_adapted": _online_loss(
                carrier, auxiliaries[1], source_code, source_previous, patch_size
            ),
            "target_base": _online_loss(
                carrier, auxiliaries[3], target_zero, target_previous, patch_size
            ),
            "target_current_one": _online_loss(
                carrier, auxiliaries[3], current_one, target_previous, patch_size
            ),
            "target_current_two": _online_loss(
                carrier, auxiliaries[3], current_two, target_previous, patch_size
            ),
            "function_before": function_before,
            "function_after": function_after,
            "untransported_before": untransported_before,
            "untransported_after": untransported_after,
            "shuffle_before": shuffled_before,
            "shuffle_after": shuffled_after,
        }
        row = {
            "scene": scene,
            "sequence": sequence,
            "source_frame": source,
            "target_frame": target,
            "frame_indices": indices,
            "valid_pixels": valid_pixels,
            "zero_code_maximum_abs_difference": zero_parity,
            "source_displacement_mean_norm": float(
                torch.linalg.vector_norm(source_displacement, dim=-1).mean()
            ),
            "mean_code_transport_distance": float(code_transport_distance.mean()),
            "mean_displacement_transport_distance": float(
                displacement_transport_distance.mean()
            ),
            "online_losses": online,
            "errors": errors,
            "alignment_scales": scales,
            "gain_vs_second_current": errors["current_two"] - errors["function_transport"],
            "gain_vs_direct_code": errors["direct_code_transport"]
            - errors["function_transport"],
            "gain_vs_untransported": errors["untransported_function"]
            - errors["function_transport"],
            "gain_vs_shuffle": errors["shuffled_function"] - errors["function_transport"],
        }
        rows.append(row)
        print(
            f"[{len(rows):02d}/16] {sequence} {source}->{target} "
            f"second={errors['current_two']:.6f} function={errors['function_transport']:.6f} "
            f"gain={row['gain_vs_second_current']:.2e} "
            f"code={row['gain_vs_direct_code']:.2e} shuffle={row['gain_vs_shuffle']:.2e}",
            flush=True,
        )
        del dataset, views, state, predictions, auxiliaries, candidate_predictions
        del source_code, source_displacement, current_one, current_two, function_code
        gc.collect()
        torch.cuda.empty_cache()

    gain_keys = (
        "gain_vs_second_current",
        "gain_vs_direct_code",
        "gain_vs_untransported",
        "gain_vs_shuffle",
    )
    scene_means = {key: _scene_means(rows, key) for key in gain_keys}
    means = {
        key: float(np.mean(list(scene_means[key].values()))) for key in gain_keys
    }
    intervals = {
        key: _stratified_interval(
            rows,
            key,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["bootstrap_seed"]) + index,
        )
        for index, key in enumerate(gain_keys)
    }
    mean_second = float(
        np.mean(
            [
                np.mean([row["errors"]["current_two"] for row in rows if row["scene"] == scene])
                for scene in sorted({row["scene"] for row in rows})
            ]
        )
    )
    relative_gain = means["gain_vs_second_current"] / max(mean_second, 1e-12)
    harm = float(np.mean([row["gain_vs_second_current"] < 0 for row in rows]))
    success = config["success"]
    gates = {
        "exact_counts": len(rows) == int(success["exact_pairs"])
        and len({row["scene"] for row in rows}) == int(success["exact_scenes"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                *row["errors"].values(),
                *row["online_losses"].values(),
                *(row[key] for key in gain_keys),
            )
        ),
        "zero_code_parity": max(row["zero_code_maximum_abs_difference"] for row in rows)
        <= float(success["maximum_zero_code_abs_difference"]),
        "source_step_descends_every_pair": all(
            row["online_losses"]["source_adapted"] < row["online_losses"]["source_base"]
            for row in rows
        ),
        "first_current_step_descends_every_pair": all(
            row["online_losses"]["target_current_one"] < row["online_losses"]["target_base"]
            for row in rows
        ),
        "second_current_step_descends_every_pair": all(
            row["online_losses"]["target_current_two"]
            < row["online_losses"]["target_current_one"]
            for row in rows
        ),
        "function_pullback_descends_every_pair": all(
            row["online_losses"]["function_after"] < row["online_losses"]["function_before"]
            for row in rows
        ),
        "function_beats_second_all_scenes": all(
            value > 0 for value in scene_means["gain_vs_second_current"].values()
        ),
        "function_beats_second_positive_ci": intervals["gain_vs_second_current"][0] > 0,
        "minimum_relative_gain": relative_gain
        >= float(success["minimum_relative_gain_vs_second_current"]),
        "harm_within_bound": harm <= float(success["maximum_harm_fraction"]),
        "function_beats_code_all_scenes": all(
            value > 0 for value in scene_means["gain_vs_direct_code"].values()
        ),
        "function_beats_code_positive_ci": intervals["gain_vs_direct_code"][0] > 0,
        "function_beats_untransported_all_scenes": all(
            value > 0 for value in scene_means["gain_vs_untransported"].values()
        ),
        "function_beats_untransported_positive_ci": intervals["gain_vs_untransported"][0] > 0,
        "function_beats_shuffle_all_scenes": all(
            value > 0 for value in scene_means["gain_vs_shuffle"].values()
        ),
        "function_beats_shuffle_positive_ci": intervals["gain_vs_shuffle"][0] > 0,
        "no_fit_or_heldout_access": True,
    }
    result = {
        "experiment": "EXP-067",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "depth_preparation_sha256": _sha256(preparation_path),
        "fit_performed": False,
        "oracle_pose_pair_selection_only": True,
        "pose_used_by_online_method": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "means": means,
        "scene_means": scene_means,
        "bootstrap_95": intervals,
        "mean_second_current_epe": mean_second,
        "relative_gain_vs_second_current": relative_gain,
        "function_transport_harm_fraction": harm,
        "maximum_zero_code_abs_difference": max(
            row["zero_code_maximum_abs_difference"] for row in rows
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
