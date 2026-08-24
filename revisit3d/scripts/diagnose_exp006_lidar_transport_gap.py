#!/usr/bin/env python3
"""LiDAR/pose context-only upper bound for EXP-006 3D atom transport.

The frozen head's depth and pose live in a per-segment VGGT gauge.  This
diagnostic projects nuScenes LiDAR into matched source/current *context* frames,
uses those sparse depths to scale the dense predicted depth into metric units,
and transports atoms in the common nuScenes world frame.  Future query
geometry, poses, LiDAR, and tracks are never accessed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import adapt_context, query_readout_loss, require_exp006_split
from revisit3d.losses import normalized_future_utility
from revisit3d.models import (
    Sim3Alignment,
    SpatialPlasticityHead,
    align_atoms,
    apply_sim3,
    backproject_tokens,
    geometry_transport,
    local_knn_scale,
    visual_transport,
)
from revisit3d.scripts.train_exp006_atom import _segments


CONDITIONS = (
    "visual",
    "predicted_geometry",
    "predicted_geometry_appearance",
    "lidar_scaled_geometry",
    "lidar_scaled_geometry_appearance",
    "lidar_sparse_nearest",
    "centered_visual",
    "centered_predicted_geometry",
    "centered_lidar_scaled_geometry",
    "centered_lidar_scaled_geometry_appearance",
)


def _transform(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    return Quaternion(rotation).rotation_matrix @ points + np.asarray(translation)[:, None]


def _inverse_transform(points: np.ndarray, rotation: list[float], translation: list[float]) -> np.ndarray:
    return Quaternion(rotation).rotation_matrix.T @ (points - np.asarray(translation)[:, None])


def _lidar_depth_grid(nusc: NuScenes, camera_token: str, side: int) -> tuple[np.ndarray, np.ndarray]:
    camera = nusc.get("sample_data", camera_token)
    sample = nusc.get("sample", camera["sample_token"])
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cloud = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar["token"]))
    points = cloud.points[:3]
    lidar_calibration = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar["ego_pose_token"])
    camera_pose = nusc.get("ego_pose", camera["ego_pose_token"])
    camera_calibration = nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
    points = _transform(points, lidar_calibration["rotation"], lidar_calibration["translation"])
    points = _transform(points, lidar_pose["rotation"], lidar_pose["translation"])
    points = _inverse_transform(points, camera_pose["rotation"], camera_pose["translation"])
    points = _inverse_transform(
        points, camera_calibration["rotation"], camera_calibration["translation"],
    )
    depth = points[2]
    intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
    pixels = intrinsic @ points
    pixels[:2] /= np.maximum(pixels[2:3], 1e-8)
    u, v = pixels[0], pixels[1]
    keep = (
        (depth > 1.0)
        & (u >= 0) & (u < camera["width"])
        & (v >= 0) & (v < camera["height"])
    )
    cell_x = np.floor(u[keep] / camera["width"] * side).astype(np.int64).clip(0, side - 1)
    cell_y = np.floor(v[keep] / camera["height"] * side).astype(np.int64).clip(0, side - 1)
    grid = np.full(side * side, np.inf, dtype=np.float32)
    np.minimum.at(grid, cell_y * side + cell_x, depth[keep].astype(np.float32))
    valid = np.isfinite(grid)
    grid[~valid] = 0
    return grid.reshape(side, side), valid.reshape(side, side)


def _segment_lidar(
    nusc: NuScenes,
    camera_tokens: dict[str, str],
    dataset: RevisitEpisodeDataset,
    scene: str,
    frame_indices: list[int],
    side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    depths, masks = [], []
    metadata = dataset._meta(scene)  # Same immutable camera metadata used by the dataset loader.
    for frame_index in frame_indices:
        absolute = Path(metadata["frames"][frame_index]["file_path"])
        try:
            relative = str(absolute.relative_to(nusc.dataroot))
        except ValueError as error:
            raise RuntimeError(f"camera path is outside nuScenes root: {absolute}") from error
        if relative not in camera_tokens:
            raise RuntimeError(f"nuScenes sample_data token missing for {relative}")
        depth, valid = _lidar_depth_grid(nusc, camera_tokens[relative], side)
        depths.append(torch.from_numpy(depth))
        masks.append(torch.from_numpy(valid))
    return torch.stack(depths), torch.stack(masks)


def _metric_depth(
    predicted_depth: torch.Tensor, lidar_depth: torch.Tensor, lidar_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    predicted = predicted_depth[0]
    lidar_depth = lidar_depth.to(device=predicted.device, dtype=predicted.dtype)
    lidar_valid = lidar_valid.to(device=predicted.device)
    usable = lidar_valid & torch.isfinite(predicted) & (predicted > 1e-5) & (lidar_depth > 1.0)
    if int(usable.sum()) < 32:
        raise RuntimeError("fewer than 32 projected LiDAR cells are available for gauge calibration")
    scale = (lidar_depth[usable] / predicted[usable]).median()
    return predicted_depth * scale, scale, int(usable.sum())


def _identity(reference: torch.Tensor, count: int) -> Sim3Alignment:
    return Sim3Alignment(
        scale=reference.new_ones(()),
        rotation=torch.eye(3, device=reference.device, dtype=reference.dtype),
        translation=reference.new_zeros(3),
        valid=True,
        correspondences=count,
        inliers=count,
        inlier_ratio=1.0,
        normalized_median_residual=0.0,
        source_rank_ratio=1.0,
        target_rank_ratio=1.0,
    )


def _sparse_nearest_code(
    source_code: torch.Tensor,
    source_xyz: torch.Tensor,
    source_valid: torch.Tensor,
    target_xyz: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    maximum_distance: float = 2.0,
) -> tuple[torch.Tensor, float]:
    shape = target_xyz.shape
    output = source_code.new_zeros(target_xyz.shape[:-1] + (source_code.shape[-1],))
    source_mask = source_valid.flatten().to(source_code.device)
    target_mask = target_valid.flatten().to(source_code.device)
    source_points = source_xyz.flatten(0, 2)[source_mask]
    target_points = target_xyz.flatten(0, 2)[target_mask]
    if not source_points.numel() or not target_points.numel():
        return output, 0.0
    distance = torch.cdist(target_points.float(), source_points.float())
    nearest_distance, nearest = distance.min(dim=-1)
    accepted = nearest_distance <= maximum_distance
    target_indices = target_mask.nonzero(as_tuple=False).flatten()[accepted]
    source_values = source_code.flatten(0, 2)[source_mask][nearest[accepted]]
    output.flatten(0, 2)[target_indices] = source_values
    coverage = float(accepted.sum() / max(target_valid.numel(), 1))
    return output.reshape(*shape[:-1], source_code.shape[-1]), coverage


def _neighbor_indices(source_xyz: torch.Tensor, target_xyz: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cdist(target_xyz.float(), source_xyz.float()).topk(k, dim=-1, largest=False).indices


def _neighbor_agreement(predicted: torch.Tensor, metric: torch.Tensor) -> tuple[float, float]:
    top1 = float((predicted[:, 0] == metric[:, 0]).float().mean())
    intersection = (predicted[:, :, None] == metric[:, None, :]).any(dim=-1).float().sum(dim=-1)
    return top1, float((intersection / metric.shape[1]).mean())


def _summarize(rows: list[dict], condition: str, epsilon: float) -> dict:
    subset = [row for row in rows if row["condition"] == condition]
    valid = [row for row in subset if row["valid"]]
    values = np.asarray([row["utility"] for row in valid], dtype=np.float64)
    return {
        "episodes": len(subset),
        "valid_rate": len(valid) / max(len(subset), 1),
        "mean_utility": float(values.mean()) if values.size else None,
        "median_utility": float(np.median(values)) if values.size else None,
        "beneficial_rate": float(np.mean(values > epsilon)) if values.size else 0.0,
        "harmful_rate": float(np.mean(values < -epsilon)) if values.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument("--nuscenes-root", default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_centered_atom_transport_train_v26.json",
    )
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 LiDAR diagnostic requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if config["data"]["split"] != "train":
        raise RuntimeError("LiDAR transport diagnosis is train-only")
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(config["stage1"]["output_checkpoint"], map_location="cpu", weights_only=False)
    if not (
        cache.get("protocol_revision") == checkpoint.get("protocol_revision") == config["protocol_revision"]
        and cache.get("split") == checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage-1 cache/checkpoint protocol mismatch")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False)
    camera_tokens = {
        row["filename"]: row["token"] for row in nusc.sample_data if row.get("channel") == "CAM_FRONT"
    }
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    epsilon = float(stage1["utility_deadband_minimum"])
    strength = float(stage1["reuse_strength"])
    appearance_weight = float(stage1["appearance_weight"])
    rows: list[dict] = []
    diagnostics: list[dict] = []

    with torch.enable_grad():
        for index, sample in enumerate(dataset):
            record = dataset.records[index]
            current, query, sources = _segments(cache, dataset.records, index, config, device)
            label, source = sources[0]
            if label != "matched_a":
                raise RuntimeError("the first registered candidate must be matched_a")
            current_zero = current.atom(head)
            source_zero = source.atom(head)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            source_code, _ = adapt_context(
                head, source, source_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            source_atom = replace(source_zero, code=source_code.detach())
            centered_source = replace(
                source_atom,
                code=source_code.detach() - source_code.detach().mean(dim=(1, 2), keepdim=True),
            )
            current_query = query_readout_loss(head, replace(current_zero, code=current_code), query)
            side = source.base_depth.shape[-1]

            source_lidar, source_valid = _segment_lidar(
                nusc, camera_tokens, dataset, record["a"]["scene"], record["a"]["frames"], side,
            )
            current_lidar, current_valid = _segment_lidar(
                nusc, camera_tokens, dataset, record["a_prime"]["scene"],
                record["a_prime"]["frames"], side,
            )
            source_metric_depth, source_scale, source_observations = _metric_depth(
                source.base_depth, source_lidar, source_valid,
            )
            current_metric_depth, current_scale, current_observations = _metric_depth(
                current.base_depth, current_lidar, current_valid,
            )
            source_w2c = sample["a"]["context"]["w2c"].unsqueeze(0).to(device)
            current_w2c = sample["a_prime"]["context"]["w2c"].unsqueeze(0).to(device)
            source_metric_xyz = backproject_tokens(
                source_metric_depth, source.intrinsics, source_w2c, image_size=source.image_size,
            )
            current_metric_xyz = backproject_tokens(
                current_metric_depth, current.intrinsics, current_w2c, image_size=current.image_size,
            )
            source_metric = replace(
                source_atom, xyz=source_metric_xyz, scale=local_knn_scale(source_metric_xyz),
            )
            centered_source_metric = replace(
                centered_source, xyz=source_metric_xyz, scale=source_metric.scale,
            )
            current_metric = replace(
                current_zero, xyz=current_metric_xyz, scale=local_knn_scale(current_metric_xyz),
            )
            identity = _identity(source_metric_xyz, source_metric_xyz.shape[1] * source_metric_xyz.shape[2])
            predicted_alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
            source_lidar_gpu = source_lidar.unsqueeze(0).to(device)
            current_lidar_gpu = current_lidar.unsqueeze(0).to(device)
            source_sparse_xyz = backproject_tokens(
                source_lidar_gpu, source.intrinsics, source_w2c, image_size=source.image_size,
            )
            current_sparse_xyz = backproject_tokens(
                current_lidar_gpu, current.intrinsics, current_w2c, image_size=current.image_size,
            )
            sparse_code, sparse_coverage = _sparse_nearest_code(
                source_code.detach(), source_sparse_xyz, source_valid.unsqueeze(0),
                current_sparse_xyz, current_valid.unsqueeze(0),
            )
            transported = {
                "visual": visual_transport(source_atom, current_zero).code,
                "predicted_geometry": geometry_transport(
                    source_atom, current_zero, [predicted_alignment], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "predicted_geometry_appearance": geometry_transport(
                    source_atom, current_zero, [predicted_alignment],
                    appearance_weight=appearance_weight, neighbors=args.neighbors,
                ).code,
                "lidar_scaled_geometry": geometry_transport(
                    source_metric, current_metric, [identity], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "lidar_scaled_geometry_appearance": geometry_transport(
                    source_metric, current_metric, [identity],
                    appearance_weight=appearance_weight, neighbors=args.neighbors,
                ).code,
                "lidar_sparse_nearest": sparse_code,
                "centered_visual": visual_transport(centered_source, current_zero).code,
                "centered_predicted_geometry": geometry_transport(
                    centered_source, current_zero, [predicted_alignment], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "centered_lidar_scaled_geometry": geometry_transport(
                    centered_source_metric, current_metric, [identity], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "centered_lidar_scaled_geometry_appearance": geometry_transport(
                    centered_source_metric, current_metric, [identity],
                    appearance_weight=appearance_weight, neighbors=args.neighbors,
                ).code,
            }
            for condition in CONDITIONS:
                valid = "predicted_geometry" not in condition or predicted_alignment.valid
                if valid:
                    candidate_code = (current_code + strength * transported[condition]).clamp(-1, 1)
                    query_loss = query_readout_loss(
                        head, replace(current_zero, code=candidate_code), query,
                    )
                    utility = normalized_future_utility(current_query, query_loss)
                    query_value, utility_value = float(query_loss.detach()), float(utility.detach())
                else:
                    query_value, utility_value = None, None
                rows.append({
                    "episode": sample["episode_id"], "condition": condition, "valid": bool(valid),
                    "query_loss": query_value, "utility": utility_value,
                })

            detail = {
                "episode": sample["episode_id"],
                "source_lidar_cells": source_observations,
                "current_lidar_cells": current_observations,
                "source_metric_per_predicted_depth": float(source_scale),
                "current_metric_per_predicted_depth": float(current_scale),
                "lidar_sparse_transport_coverage": sparse_coverage,
                "predicted_alignment_valid": predicted_alignment.valid,
            }
            if predicted_alignment.valid:
                predicted_neighbors = _neighbor_indices(
                    apply_sim3(source_atom.xyz[0].flatten(0, 1), predicted_alignment),
                    current_zero.xyz[0].flatten(0, 1), args.neighbors,
                )
                metric_neighbors = _neighbor_indices(
                    source_metric_xyz[0].flatten(0, 1), current_metric_xyz[0].flatten(0, 1),
                    args.neighbors,
                )
                top1, recall = _neighbor_agreement(predicted_neighbors, metric_neighbors)
                detail["predicted_vs_lidar_scaled_top1_agreement"] = top1
                detail[f"predicted_vs_lidar_scaled_recall_at_{args.neighbors}"] = recall
            else:
                detail["predicted_vs_lidar_scaled_top1_agreement"] = None
                detail[f"predicted_vs_lidar_scaled_recall_at_{args.neighbors}"] = None
            diagnostics.append(detail)
            print(json.dumps(detail), flush=True)

    summary = {condition: _summarize(rows, condition, epsilon) for condition in CONDITIONS}
    aligned = [row for row in diagnostics if row["predicted_alignment_valid"]]
    agreement = {
        "predicted_alignment_valid_rate": len(aligned) / max(len(diagnostics), 1),
        "mean_top1_agreement": float(np.mean([
            row["predicted_vs_lidar_scaled_top1_agreement"] for row in aligned
        ])) if aligned else None,
        f"mean_recall_at_{args.neighbors}": float(np.mean([
            row[f"predicted_vs_lidar_scaled_recall_at_{args.neighbors}"] for row in aligned
        ])) if aligned else None,
        "mean_sparse_transport_coverage": float(np.mean([
            row["lidar_sparse_transport_coverage"] for row in diagnostics
        ])),
    }
    payload = {
        "experiment": "EXP-006", "stage": "stage1_centered_atom_transport_diagnostic",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "oracle_context_geometry_diagnostic_only": True,
        "oracle_geometry_scope": "matched_source_and_current_context_only",
        "query_geometry_accessed": False, "validation_accessed": False,
        "depth_gauge_calibration": "segment_median_projected_lidar_depth_ratio",
        "neighbors": args.neighbors, "reuse_strength": strength,
        "summary": summary, "neighbor_agreement": agreement,
        "diagnostics": diagnostics, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary, **agreement}), flush=True)


if __name__ == "__main__":
    main()
