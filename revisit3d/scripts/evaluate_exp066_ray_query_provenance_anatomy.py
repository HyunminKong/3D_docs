#!/usr/bin/env python3
"""Zero-fit causal evidence-provenance anatomy for EXP-066."""
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
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import _model_view
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / max(values.size - 1, 1)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rank(first)
    second_rank = _rank(second)
    if first_rank.std() <= 1e-12 or second_rank.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _aurc(risk: np.ndarray, error: np.ndarray) -> float:
    """Area under selective risk-coverage curve; lower risk is retained first."""
    order = np.argsort(np.asarray(risk), kind="mergesort")
    ordered_error = np.asarray(error, dtype=np.float64)[order]
    return float(np.mean(np.cumsum(ordered_error) / np.arange(1, len(order) + 1)))


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1
    )
    return homogeneous @ transform.T[..., :3]


def _historical_support(
    query_world: np.ndarray,
    history_views: list[dict],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> np.ndarray:
    flat_world = query_world.reshape(-1, 3)
    support = np.zeros(len(flat_world), dtype=bool)
    finite_world = np.isfinite(flat_world).all(axis=-1)
    for view in history_views:
        camera = _transform(
            flat_world, np.linalg.inv(np.asarray(view["camera_pose"], dtype=np.float64))
        )
        z = camera[:, 2]
        intrinsics = np.asarray(view["camera_intrinsics"], dtype=np.float64)
        u = intrinsics[0, 0] * camera[:, 0] / np.maximum(z, 1e-12) + intrinsics[0, 2]
        v = intrinsics[1, 1] * camera[:, 1] / np.maximum(z, 1e-12) + intrinsics[1, 2]
        depth = np.asarray(view["depthmap"], dtype=np.float64)
        height, width = depth.shape
        x = np.rint(u).astype(np.int64)
        y = np.rint(v).astype(np.int64)
        inside = finite_world & (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        measured = np.zeros_like(z)
        measured[inside] = depth[y[inside], x[inside]]
        tolerance = np.maximum(absolute_tolerance, relative_tolerance * z)
        support |= inside & (measured > 0) & (np.abs(measured - z) <= tolerance)
    return support.reshape(query_world.shape[:-1])


def _ray_query(
    carrier: FrozenCUT3RCarrier,
    state,
    first_view: dict,
    query_view: dict,
    get_ray_map,
    inference_step,
) -> dict[str, torch.Tensor]:
    height, width = np.asarray(query_view["depthmap"]).shape
    ray_map = get_ray_map(
        np.asarray(first_view["camera_pose"], dtype=np.float64),
        np.asarray(query_view["camera_pose"], dtype=np.float64),
        np.asarray(query_view["camera_intrinsics"], dtype=np.float64),
        height,
        width,
    ).astype(np.float32)
    view = {
        "img": torch.full((1, 3, height, width), float("nan"), dtype=torch.float32),
        "ray_map": torch.from_numpy(ray_map).unsqueeze(0),
        "true_shape": torch.tensor([[height, width]], dtype=torch.int64),
        "idx": 4,
        "instance": str(query_view["instance"]),
        "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
        "img_mask": torch.tensor([False]),
        "ray_mask": torch.tensor([True]),
        "update": torch.tensor([False]),
        "reset": torch.tensor([False]),
    }
    state_args = (
        state.state_feat,
        state.state_pos,
        state.init_state_feat,
        state.mem,
        state.init_mem,
    )
    return inference_step(view, state_args, carrier.model, "cuda", False)["pred"]


def _context_metrics(
    prediction: dict[str, torch.Tensor],
    history_predictions: list[dict[str, torch.Tensor]],
    views: list[dict],
    patch_size: int,
    minimum_depth: float,
    maximum_depth: float,
    minimum_valid: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    offset = patch_size // 2
    query_world = np.asarray(views[-1]["pts3d"], dtype=np.float64)[
        offset::patch_size, offset::patch_size
    ]
    query_depth = np.asarray(views[-1]["depthmap"], dtype=np.float64)[
        offset::patch_size, offset::patch_size
    ]
    query_valid = np.asarray(views[-1]["valid_mask"], dtype=bool)[
        offset::patch_size, offset::patch_size
    ]
    target_first = _transform(
        query_world,
        np.linalg.inv(np.asarray(views[0]["camera_pose"], dtype=np.float64)),
    )
    predicted_query = (
        prediction["pts3d_in_other_view"][0, offset::patch_size, offset::patch_size]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    confidence = (
        prediction["conf"][0, offset::patch_size, offset::patch_size]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    supported = _historical_support(
        query_world,
        views[:-1],
        absolute_tolerance,
        relative_tolerance,
    )
    valid = (
        query_valid
        & np.isfinite(target_first).all(axis=-1)
        & np.isfinite(predicted_query).all(axis=-1)
        & np.isfinite(confidence)
        & (query_depth >= minimum_depth)
        & (query_depth <= maximum_depth)
    )
    if int(valid.sum()) < minimum_valid:
        raise RuntimeError("EXP-066 has insufficient valid query patches")

    target = target_first[valid]
    predicted = predicted_query[valid]
    depth = query_depth[valid]
    target_range = np.linalg.norm(target, axis=-1)
    predicted_range = np.linalg.norm(predicted, axis=-1)
    scale = float(
        np.median(target_range[predicted_range > 1e-8])
        / np.median(predicted_range[predicted_range > 1e-8])
    )
    error = np.linalg.norm(scale * predicted - target, axis=-1) / np.maximum(
        depth, minimum_depth
    )

    history_points = []
    for history_prediction in history_predictions:
        points = (
            history_prediction["pts3d_in_other_view"][
                0, offset::patch_size, offset::patch_size
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
            .reshape(-1, 3)
        )
        history_points.append(points[np.isfinite(points).all(axis=-1)])
    history = np.concatenate(history_points, axis=0)
    query_tensor = torch.from_numpy(predicted).cuda().float().unsqueeze(0)
    history_tensor = torch.from_numpy(history).cuda().float().unsqueeze(0)
    nearest = (
        torch.cdist(query_tensor, history_tensor).amin(dim=-1)[0].cpu().numpy().astype(np.float64)
    )
    provenance_risk = nearest / np.maximum(predicted_range, 1e-6)
    confidence_risk = -confidence[valid]
    combined_risk = 0.5 * (_rank(provenance_risk) + _rank(confidence_risk))
    support_valid = supported[valid]
    if not support_valid.any() or support_valid.all():
        raise RuntimeError("EXP-066 context has no supported/unsupported split")

    supported_error = float(error[support_valid].mean())
    unsupported_error = float(error[~support_valid].mean())
    rho_provenance = _spearman(provenance_risk, error)
    rho_confidence = _spearman(confidence_risk, error)
    aurc_confidence = _aurc(confidence_risk, error)
    aurc_provenance = _aurc(provenance_risk, error)
    aurc_combined = _aurc(combined_risk, error)
    metrics = {
        "valid_patches": int(valid.sum()),
        "supported_patches": int(support_valid.sum()),
        "unsupported_patches": int((~support_valid).sum()),
        "supported_fraction": float(support_valid.mean()),
        "unsupported_fraction": float((~support_valid).mean()),
        "supported_epe": supported_error,
        "unsupported_epe": unsupported_error,
        "error_gap": unsupported_error - supported_error,
        "relative_error_gap": (unsupported_error - supported_error) / max(supported_error, 1e-12),
        "provenance_error_spearman": rho_provenance,
        "confidence_error_spearman": rho_confidence,
        "spearman_advantage": rho_provenance - rho_confidence,
        "confidence_aurc": aurc_confidence,
        "provenance_aurc": aurc_provenance,
        "combined_aurc": aurc_combined,
        "combined_aurc_gain": aurc_confidence - aurc_combined,
        "relative_combined_aurc_gain": (aurc_confidence - aurc_combined)
        / max(aurc_confidence, 1e-12),
        "metric_scale": scale,
    }
    arrays = {
        "error": error,
        "support": support_valid,
        "provenance_risk": provenance_risk,
        "confidence_risk": confidence_risk,
        "combined_risk": combined_risk,
    }
    return metrics, arrays


def _scene_summary(rows: list[dict], arrays: list[dict[str, np.ndarray]]) -> dict[str, dict]:
    summaries = {}
    for scene in sorted({row["scene"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["scene"] == scene]
        joined = {
            key: np.concatenate([arrays[index][key] for index in indices])
            for key in arrays[0]
        }
        support = joined["support"]
        error = joined["error"]
        confidence_aurc = _aurc(joined["confidence_risk"], error)
        combined_aurc = _aurc(joined["combined_risk"], error)
        summaries[scene] = {
            "valid_patches": int(error.size),
            "supported_fraction": float(support.mean()),
            "unsupported_fraction": float((~support).mean()),
            "supported_epe": float(error[support].mean()),
            "unsupported_epe": float(error[~support].mean()),
            "error_gap": float(error[~support].mean() - error[support].mean()),
            "provenance_error_spearman": _spearman(joined["provenance_risk"], error),
            "confidence_error_spearman": _spearman(joined["confidence_risk"], error),
            "confidence_aurc": confidence_aurc,
            "combined_aurc": combined_aurc,
            "combined_aurc_gain": confidence_aurc - combined_aurc,
        }
    return summaries


def _bootstrap_interval(rows: list[dict], key: str, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    scenes = sorted({row["scene"] for row in rows})
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        scene_values = []
        for scene in scenes:
            scene_rows = by_scene[scene]
            chosen = rng.integers(0, len(scene_rows), size=len(scene_rows))
            scene_values.append(np.mean([scene_rows[index][key] for index in chosen]))
        draws[sample] = np.mean(scene_values)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-066_ray_query_provenance_anatomy_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-ray-provenance", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_ray_provenance or not torch.cuda.is_available():
        raise SystemExit("EXP-066 requires train-RGB-D provenance confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-066 result already exists")
    if not preparation_path.exists():
        raise RuntimeError("EXP-066 selected train depths have not been registered")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    expected_frames = len(config["data"]["sequences"]) * len(
        config["data"]["query_frames"]
    ) * len(config["data"]["frame_offsets"])
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and preparation["experiment"] == "EXP-066"
        and preparation["selected_frames"] == expected_frames
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
        and int(config["data"]["history_views"]) == 4
        and list(config["data"]["frame_offsets"])[-1] == 0
    ):
        raise RuntimeError("EXP-066 source-safe contract failed")

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
    from dust3r.datasets.base.base_multiview_dataset import get_ray_map
    from dust3r.inference import inference_step
    from eval.mv_recon.data import SevenScenes

    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=8,
        basis_seed=seed,
        update_type=config["carrier"]["mode"],
    ).cuda()
    carrier.eval().requires_grad_(False)
    patch_size = int(config["carrier"]["patch_size"])
    rows: list[dict] = []
    array_rows: list[dict[str, np.ndarray]] = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for query_frame in config["data"]["query_frames"]:
            frame_indices = [
                int(query_frame) + int(offset) for offset in config["data"]["frame_offsets"]
            ]
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
            history_predictions = []
            with torch.no_grad():
                for index in range(4):
                    prediction, state, _ = carrier.step(_model_view(views[index], index), state)
                    history_predictions.append(prediction)
                query_prediction = _ray_query(
                    carrier, state, views[0], views[-1], get_ray_map, inference_step
                )
                replay_prediction = _ray_query(
                    carrier, state, views[0], views[-1], get_ray_map, inference_step
                )
            replay_difference = max(
                float(
                    (
                        query_prediction[key].detach().float()
                        - replay_prediction[key].detach().float()
                    ).abs().max()
                )
                for key in ("pts3d_in_other_view", "pts3d_in_self_view", "conf")
            )
            metrics, arrays = _context_metrics(
                query_prediction,
                history_predictions,
                views,
                patch_size,
                float(config["metric"]["minimum_depth_m"]),
                float(config["metric"]["maximum_depth_m"]),
                int(config["metric"]["minimum_valid_patches"]),
                float(config["support"]["absolute_depth_tolerance_m"]),
                float(config["support"]["relative_depth_tolerance"]),
            )
            row = {
                "scene": scene,
                "sequence": sequence,
                "query_frame": int(query_frame),
                "frame_indices": frame_indices,
                "history_rgb_views": 4,
                "query_rgb_supplied": False,
                "query_update": False,
                "replay_maximum_abs_difference": replay_difference,
                **metrics,
            }
            rows.append(row)
            array_rows.append(arrays)
            print(
                f"[{len(rows):02d}/16] {sequence}:{query_frame} "
                f"support={metrics['supported_fraction']:.1%} "
                f"gap={metrics['relative_error_gap']:.1%} "
                f"rho={metrics['provenance_error_spearman']:.3f}/"
                f"{metrics['confidence_error_spearman']:.3f} "
                f"aurc_gain={metrics['relative_combined_aurc_gain']:.1%}",
                flush=True,
            )
            del dataset, views, state, history_predictions, query_prediction, replay_prediction
            gc.collect()
            torch.cuda.empty_cache()

    scene = _scene_summary(rows, array_rows)
    all_arrays = {
        key: np.concatenate([arrays[key] for arrays in array_rows]) for key in array_rows[0]
    }
    support = all_arrays["support"]
    error = all_arrays["error"]
    aggregate_confidence_aurc = _aurc(all_arrays["confidence_risk"], error)
    aggregate_combined_aurc = _aurc(all_arrays["combined_risk"], error)
    aggregate = {
        "valid_patches": int(error.size),
        "supported_fraction": float(support.mean()),
        "unsupported_fraction": float((~support).mean()),
        "supported_epe": float(error[support].mean()),
        "unsupported_epe": float(error[~support].mean()),
        "error_gap": float(error[~support].mean() - error[support].mean()),
        "relative_error_gap": float(
            (error[~support].mean() - error[support].mean()) / error[support].mean()
        ),
        "provenance_error_spearman": _spearman(all_arrays["provenance_risk"], error),
        "confidence_error_spearman": _spearman(all_arrays["confidence_risk"], error),
        "spearman_advantage": _spearman(all_arrays["provenance_risk"], error)
        - _spearman(all_arrays["confidence_risk"], error),
        "confidence_aurc": aggregate_confidence_aurc,
        "provenance_aurc": _aurc(all_arrays["provenance_risk"], error),
        "combined_aurc": aggregate_combined_aurc,
        "combined_aurc_gain": aggregate_confidence_aurc - aggregate_combined_aurc,
        "relative_combined_aurc_gain": (
            aggregate_confidence_aurc - aggregate_combined_aurc
        ) / aggregate_confidence_aurc,
        "maximum_replay_abs_difference": max(
            row["replay_maximum_abs_difference"] for row in rows
        ),
    }
    samples = int(config["statistics"]["bootstrap_samples"])
    intervals = {
        "error_gap": _bootstrap_interval(rows, "error_gap", samples, seed + 1),
        "spearman_advantage": _bootstrap_interval(rows, "spearman_advantage", samples, seed + 2),
        "combined_aurc_gain": _bootstrap_interval(rows, "combined_aurc_gain", samples, seed + 3),
    }
    success = config["success"]
    gates = {
        "exact_counts": len(rows) == int(success["exact_contexts"])
        and len(scene) == int(success["exact_scenes"]),
        "replay_within_tolerance": aggregate["maximum_replay_abs_difference"]
        <= float(success["maximum_replay_abs_difference"]),
        "support_coverage_all_scenes": all(
            item["supported_fraction"] >= float(success["minimum_scene_supported_fraction"])
            and item["unsupported_fraction"] >= float(success["minimum_scene_unsupported_fraction"])
            for item in scene.values()
        ),
        "error_gap_positive_all_scenes": all(item["error_gap"] > 0 for item in scene.values()),
        "error_gap_bootstrap_lower_positive": intervals["error_gap"][0] > 0,
        "minimum_relative_error_gap": aggregate["relative_error_gap"]
        >= float(success["minimum_relative_error_gap"]),
        "provenance_spearman_positive_all_scenes": all(
            item["provenance_error_spearman"] > 0 for item in scene.values()
        ),
        "minimum_provenance_spearman": aggregate["provenance_error_spearman"]
        >= float(success["minimum_provenance_error_spearman"]),
        "minimum_spearman_advantage": aggregate["spearman_advantage"]
        >= float(success["minimum_spearman_advantage_over_confidence"]),
        "spearman_advantage_bootstrap_lower_positive": intervals["spearman_advantage"][0] > 0,
        "combined_aurc_gain_positive_all_scenes": all(
            item["combined_aurc_gain"] > 0 for item in scene.values()
        ),
        "combined_aurc_gain_bootstrap_lower_positive": intervals["combined_aurc_gain"][0] > 0,
        "minimum_relative_combined_aurc_gain": aggregate["relative_combined_aurc_gain"]
        >= float(success["minimum_relative_aurc_gain"]),
    }
    result = {
        "experiment": "EXP-066",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "depth_preparation_sha256": _sha256(preparation_path),
        "validation_accessed": False,
        "terminal_accessed": False,
        "fit_performed": False,
        "query_rgb_supplied": False,
        "query_update": False,
        "oracle_query_camera_rays": True,
        "summary": aggregate,
        "scene_summary": scene,
        "bootstrap_95": intervals,
        "gates": gates,
        "passed": all(gates.values()),
        "peak_gpu_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "contexts": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "contexts"}, indent=2))


if __name__ == "__main__":
    main()
