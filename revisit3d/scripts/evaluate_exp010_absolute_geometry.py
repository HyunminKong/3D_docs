#!/usr/bin/env python3
"""Post-lock sparse-LiDAR metric bridge for the frozen EXP-009 model.

The evaluator replays the exact causal reservoir-64 policy and only then reads
held-out query LiDAR for metrics.  LiDAR never enters adaptation, retrieval,
transport, routing, or retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from revisit3d.experiments import CachedAtomSegment, adapt_context, geometry_objective, observable_router_features, query_readout_loss
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _cpu_atom, _device_atom, _float_stats, _identifier, _tensor_stats, _timestamp,
)
from revisit3d.scripts.evaluate_exp009_locked_validation import _mips_score, _sha256


def _table(root: Path, name: str) -> dict[str, dict]:
    rows = json.loads((root / "v1.0-trainval" / f"{name}.json").read_text())
    return {row["token"]: row for row in rows}


def _rotation(quaternion: list[float]) -> np.ndarray:
    """nuScenes [w,x,y,z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _forward(points: np.ndarray, record: dict) -> np.ndarray:
    return points @ _rotation(record["rotation"]).T + np.asarray(record["translation"], dtype=np.float64)


def _inverse(points: np.ndarray, record: dict) -> np.ndarray:
    return (points - np.asarray(record["translation"], dtype=np.float64)) @ _rotation(record["rotation"])


class LidarProjector:
    def __init__(self, root: str | Path, *, minimum_depth: float, maximum_depth: float) -> None:
        self.root = Path(root)
        self.minimum_depth = float(minimum_depth)
        self.maximum_depth = float(maximum_depth)
        self.sample_data = _table(self.root, "sample_data")
        self.samples = _table(self.root, "sample")
        self.calibrations = _table(self.root, "calibrated_sensor")
        self.sensors = _table(self.root, "sensor")
        self.poses = _table(self.root, "ego_pose")
        self.data_by_sample_channel = {
            (
                row["sample_token"],
                self.sensors[self.calibrations[row["calibrated_sensor_token"]]["sensor_token"]]["channel"],
            ): row
            for row in self.sample_data.values()
        }
        self.camera_by_filename = {
            row["filename"]: row for row in self.sample_data.values()
            if self.sensors[self.calibrations[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            == "CAM_FRONT"
        }

    def _relative(self, path: str | Path) -> str:
        raw = Path(path)
        try:
            return str(raw.relative_to(self.root))
        except ValueError:
            value = str(raw)
            for prefix in ("samples/", "sweeps/"):
                position = value.find(prefix)
                if position >= 0:
                    return value[position:]
        raise RuntimeError(f"camera path is outside nuScenes: {path}")

    def depth_grid(self, camera_path: str | Path, side: int) -> tuple[np.ndarray, np.ndarray]:
        relative = self._relative(camera_path)
        if relative not in self.camera_by_filename:
            raise RuntimeError(f"CAM_FRONT sample_data missing for {relative}")
        camera = self.camera_by_filename[relative]
        lidar = self.data_by_sample_channel[(camera["sample_token"], "LIDAR_TOP")]
        raw = np.fromfile(self.root / lidar["filename"], dtype=np.float32)
        if raw.size % 5:
            raise RuntimeError(f"invalid nuScenes LiDAR file {lidar['filename']}")
        points = raw.reshape(-1, 5)[:, :3].astype(np.float64)
        points = _forward(points, self.calibrations[lidar["calibrated_sensor_token"]])
        points = _forward(points, self.poses[lidar["ego_pose_token"]])
        points = _inverse(points, self.poses[camera["ego_pose_token"]])
        camera_calibration = self.calibrations[camera["calibrated_sensor_token"]]
        points = _inverse(points, camera_calibration)
        depth = points[:, 2]
        intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
        pixels = points @ intrinsic.T
        u = pixels[:, 0] / np.maximum(pixels[:, 2], 1e-8)
        v = pixels[:, 1] / np.maximum(pixels[:, 2], 1e-8)
        keep = (
            (depth >= self.minimum_depth) & (depth <= self.maximum_depth)
            & (u >= 0) & (u < int(camera["width"]))
            & (v >= 0) & (v < int(camera["height"]))
        )
        cell_x = np.floor(u[keep] / int(camera["width"]) * side).astype(np.int64).clip(0, side - 1)
        cell_y = np.floor(v[keep] / int(camera["height"]) * side).astype(np.int64).clip(0, side - 1)
        grid = np.full(side * side, np.inf, dtype=np.float64)
        np.minimum.at(grid, cell_y * side + cell_x, depth[keep])
        valid = np.isfinite(grid)
        grid[~valid] = 0.0
        return grid.reshape(side, side), valid.reshape(side, side)


def _query_lidar(
    projector: LidarProjector, scene_root: Path, segment: dict, side: int,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((scene_root / segment["scene"] / "opencv_cameras.json").read_text())
    depths, masks = [], []
    for index in segment["query_frames"]:
        frame = metadata["frames"][int(index)]
        depth, mask = projector.depth_grid(scene_root / segment["scene"] / frame["file_path"], side)
        depths.append(depth)
        masks.append(mask)
    return np.stack(depths), np.stack(masks)


def _depth_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_size: tuple[int, int],
    minimum_cells: int,
) -> dict | None:
    view_rows = []
    views, height, width = prediction.shape
    image_height, image_width = image_size
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixel_x = (xx + 0.5) * image_width / width
    pixel_y = (yy + 0.5) * image_height / height
    for view in range(views):
        mask = valid[view] & np.isfinite(prediction[view]) & (prediction[view] > 1e-6)
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].astype(np.float64)
        gt = target[view][mask].astype(np.float64)
        log_error = np.log(pred) - np.log(gt)
        silog = 100.0 * np.sqrt(max(float(np.mean(log_error ** 2) - np.mean(log_error) ** 2), 0.0))
        scale = float(np.median(gt / pred))
        aligned = np.clip(pred * scale, 1e-6, None)
        ratio = np.maximum(aligned / gt, gt / aligned)
        fx, fy, cx, cy = [float(value) for value in intrinsics[view]]
        ray_norm = np.sqrt(
            ((pixel_x[mask] - cx) / fx) ** 2
            + ((pixel_y[mask] - cy) / fy) ** 2 + 1.0
        )
        view_rows.append({
            "cells": int(mask.sum()), "scale": scale, "silog": silog,
            "abs_rel": float(np.mean(np.abs(aligned - gt) / gt)),
            "rmse_m": float(np.sqrt(np.mean((aligned - gt) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "point_epe_m": float(np.mean(np.abs(aligned - gt) * ray_norm)),
        })
    if not view_rows:
        return None
    keys = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
    return {
        "valid_views": len(view_rows), "valid_cells": sum(row["cells"] for row in view_rows),
        **{key: float(np.mean([row[key] for row in view_rows])) for key in keys},
    }


def _summary(rows: list[dict], policy: str) -> dict:
    keys = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
    return {
        "targets": len(rows),
        "mean_valid_views": float(np.mean([row[policy]["valid_views"] for row in rows])),
        "mean_valid_cells": float(np.mean([row[policy]["valid_cells"] for row in rows])),
        **{key: float(np.mean([row[policy][key] for row in rows])) for key in keys},
    }


def _component_bootstrap(
    rows: list[dict], metric: str, *, samples: int, seed: int,
) -> dict:
    by_component: dict[str, list[float]] = {}
    for row in rows:
        improvement = row["current"][metric] - row["full"][metric]
        by_component.setdefault(row["component"], []).append(float(improvement))
    values = np.asarray([np.mean(by_component[key]) for key in sorted(by_component)], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": "current_error_minus_full_error", "components": len(values),
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-010_paper_geometry_v10.yaml")
    parser.add_argument("--confirm-post-lock-metric-audit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_post_lock_metric_audit:
        raise SystemExit("refusing held-out LiDAR audit without explicit confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-010 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-010 Stage-A result already exists")
    locked_config = yaml.safe_load(Path(config["locked_model"]["config"]).read_text())
    locked_result = json.loads(Path(config["locked_model"]["result"]).read_text())
    artifact_path = Path(config["locked_model"]["artifact"])
    artifact = joblib.load(artifact_path)
    if not (
        _sha256(artifact_path) == config["locked_model"]["artifact_sha256"]
        == locked_result["artifact_sha256"]
        and locked_result["terminal_no_further_test_tuning"] is True
        and locked_result["registered_gate"]["passed"] is True
        and artifact.get("test_accessed") is False
        and artifact.get("query_or_future_router_input") is False
        and int(artifact.get("bank_capacity")) == 64
    ):
        raise RuntimeError("EXP-009 frozen model contract changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 117
        and geometry.get("split") == config["data"]["split"] == "test"
        and geometry.get("pca_fit_split") == "train"
    ):
        raise RuntimeError("EXP-010 frozen cache contract failed")

    context_info, targets = {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context")):
            segment = row[tag]
            key = _identifier(segment)
            context_info.setdefault(key, {
                "id": key, "segment": segment, "cache_index": index,
                "cache_tag": cache_tag, "location": row["location"],
            })
        key = _identifier(row["a_prime"])
        target = {
            "id": key, "cache_index": index, "episode": f"target-{key}",
            "component": f"component-{int(row['component_id'])}", "location": row["location"],
            "segment": row["a_prime"],
        }
        if key in targets and targets[key]["segment"]["query_frames"] != target["segment"]["query_frames"]:
            raise RuntimeError("duplicate target has inconsistent query frames")
        targets.setdefault(key, target)
    if len(context_info) != 256 or len(targets) != 104:
        raise RuntimeError("EXP-010 target inventory changed")
    scene_root = Path(config["data"]["scene_root"])
    metadata_cache = {}
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)

    checkpoint = torch.load(locked_config["stage1"]["source_checkpoint"], map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("plasticity head is not frozen EXP-009 head")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(locked_config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    projector = LidarProjector(
        config["data"]["nuscenes_root"],
        minimum_depth=float(config["lidar"]["minimum_depth_m"]),
        maximum_depth=float(config["lidar"]["maximum_depth_m"]),
    )
    capacity = int(locked_config["bank"]["capacity"])
    candidate_count = int(locked_config["bank"]["candidate_count"])
    strength = float(locked_config["stage1"]["reuse_strength"])
    router, columns = artifact["router_model"], artifact["router_feature_columns"]
    threshold, compiled = float(artifact["router_threshold"]), artifact["utility_mips"]
    metric_rows, replay_utilities = [], []

    with torch.enable_grad():
        for location in sorted({row["location"] for row in context_info.values()}):
            events = sorted(
                [row for row in context_info.values() if row["location"] == location],
                key=lambda row: (row["timestamp"], row["id"]),
            )
            memory, bank = {}, []
            seen = 0
            generator = random.Random(
                int(locked_config["seed"]) + int(hashlib.sha1(location.encode()).hexdigest()[:8], 16)
            )
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code,
                    step_size=float(locked_config["stage1"]["ttt_step_size"]),
                    steps=int(locked_config["stage1"]["ttt_steps"]),
                )
                pre, pre_stats = geometry_objective(head, segment, zero.code, return_stats=True)
                post, post_stats = geometry_objective(head, segment, code, return_stats=True)
                state = {
                    "atom": _cpu_atom(replace(zero, code=code.detach())),
                    "descriptor": zero.key.mean(dim=(1, 2))[0].detach().cpu(),
                    "pre": float(pre.detach()), "post": float(post.detach()),
                    "pre_stats": _float_stats(pre_stats), "post_stats": _float_stats(post_stats),
                }
                if key in targets:
                    target = targets[key]
                    ranked = sorted((
                        (candidate, _mips_score(compiled, state["descriptor"], memory[candidate]["descriptor"]))
                        for candidate in bank
                    ), key=lambda row: (-row[1], row[0]))[:candidate_count]
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    query_zero = query.atom(head)
                    current_atom = replace(zero, code=code)
                    current_query_loss = query_readout_loss(head, current_atom, query)
                    evaluated = []
                    for candidate, _ in ranked:
                        source_state = memory[candidate]
                        source_atom = _device_atom(source_state["atom"], device)
                        alignment = align_atoms(source_atom.detach(), zero.detach())[0]
                        visual = visual_transport(source_atom, zero)
                        candidate_code = (code + strength * visual.code).clamp(-1, 1)
                        candidate_objective = geometry_objective(head, segment, candidate_code)
                        candidate_query_loss = query_readout_loss(head, replace(zero, code=candidate_code), query)
                        features = observable_router_features(
                            current_descriptor=zero.key.mean(dim=(1, 2))[0],
                            source_descriptor=source_atom.key.mean(dim=(1, 2))[0],
                            current_code=code, transported_code=visual.code, visual_result=visual,
                            alignment=alignment, current_pre_objective=pre, current_post_objective=post,
                            candidate_objective=candidate_objective,
                            source_pre_objective=torch.tensor(source_state["pre"], device=device),
                            source_post_objective=torch.tensor(source_state["post"], device=device),
                            current_pre_stats=pre_stats, current_post_stats=post_stats,
                            source_pre_stats=_tensor_stats(source_state["pre_stats"], device),
                            source_post_stats=_tensor_stats(source_state["post_stats"], device),
                        )
                        prediction = float(router.predict(
                            np.asarray(features.detach().cpu(), dtype=np.float64)[None, columns]
                        )[0])
                        evaluated.append((prediction, candidate_code, candidate_query_loss))
                    full_code = code
                    accepted = False
                    selected_prediction = None
                    selected_query_loss = current_query_loss
                    if evaluated:
                        selected_prediction, selected_code, selected_query_loss = max(evaluated, key=lambda row: row[0])
                        accepted = selected_prediction > threshold
                        if accepted:
                            full_code = selected_code
                        else:
                            selected_query_loss = current_query_loss
                    replay_utilities.append(float(
                        normalized_future_utility(current_query_loss, selected_query_loss).detach()
                    ))
                    base_depth = query.base_depth[0].detach().cpu().numpy()
                    current_query_code = visual_transport(current_atom, query_zero).code
                    full_query_code = visual_transport(replace(zero, code=full_code), query_zero).code
                    current_depth = head.depth(query.features, query.base_depth, current_query_code)[0, :, :, 0]
                    full_depth = head.depth(query.features, query.base_depth, full_query_code)[0, :, :, 0]
                    side = base_depth.shape[-1]
                    lidar_depth, lidar_valid = _query_lidar(projector, scene_root, target["segment"], side)
                    intrinsics = query.intrinsics[0].detach().cpu().numpy()
                    predictions = {
                        "base": base_depth,
                        "current": current_depth.detach().cpu().numpy().reshape(base_depth.shape),
                        "full": full_depth.detach().cpu().numpy().reshape(base_depth.shape),
                    }
                    metrics = {
                        policy: _depth_metrics(
                            prediction, lidar_depth, lidar_valid, intrinsics,
                            image_size=query.image_size,
                            minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
                        ) for policy, prediction in predictions.items()
                    }
                    if all(value is not None for value in metrics.values()):
                        metric_rows.append({
                            "episode": target["episode"], "component": target["component"],
                            "location": target["location"], "accepted": bool(accepted),
                            "predicted_utility": selected_prediction,
                            **metrics,
                        })
                memory[key] = state
                seen += 1
                if len(bank) < capacity:
                    bank.append(key)
                else:
                    replacement = generator.randrange(seen)
                    if replacement < capacity:
                        bank[replacement] = key
            print(json.dumps({"location": location, "metric_targets": len(metric_rows)}), flush=True)

    replay_mean = float(np.mean(replay_utilities))
    expected_replay = float(locked_result["metrics"]["reservoir_capacity64"]["router"]["mean_selected_utility"])
    if abs(replay_mean - expected_replay) > 1e-7:
        raise RuntimeError(f"locked policy replay changed: {replay_mean} vs {expected_replay}")
    summaries = {policy: _summary(metric_rows, policy) for policy in ("base", "current", "full")}
    primary_metrics = ("silog", "abs_rel", "point_epe_m")
    bootstrap = {
        metric: _component_bootstrap(
            metric_rows, metric, samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + index,
        ) for index, metric in enumerate(primary_metrics)
    }
    current_degradation = float(np.mean([
        row["current"]["abs_rel"] > row["base"]["abs_rel"] * (1 + float(config["success"]["maximum_relative_absrel_degradation"]))
        for row in metric_rows
    ]))
    full_degradation = float(np.mean([
        row["full"]["abs_rel"] > row["current"]["abs_rel"] * (1 + float(config["success"]["maximum_relative_absrel_degradation"]))
        for row in metric_rows
    ]))
    components = len({row["component"] for row in metric_rows})
    checks = {
        "component_coverage": components >= int(config["success"]["minimum_components"]),
        "target_coverage": len(metric_rows) >= int(config["success"]["minimum_targets"]),
        "silog_not_worse": summaries["full"]["silog"] <= summaries["current"]["silog"],
        "absrel_not_worse": summaries["full"]["abs_rel"] <= summaries["current"]["abs_rel"],
        "point_epe_not_worse": summaries["full"]["point_epe_m"] <= summaries["current"]["point_epe_m"],
        "one_primary_interval_positive": any(bootstrap[key]["ci95"][0] > 0 for key in primary_metrics),
        "target_degradation_not_increased": full_degradation <= current_degradation,
    }
    result = {
        "experiment": "EXP-010", "stage": "stageA_absolute_geometry_test",
        "protocol_revision": config["protocol_revision"], "split": "test",
        "post_lock_secondary_endpoint": True, "method_or_threshold_changed": False,
        "query_lidar_evaluation_only": True, "query_or_future_online_input": False,
        "config": str(config_path), "locked_artifact_sha256": _sha256(artifact_path),
        "targets": len(metric_rows), "components": components,
        "locked_proxy_utility_replay": replay_mean, "summaries": summaries,
        "bootstrap": bootstrap,
        "relative_absrel_degradation_fraction": {
            "current_over_base": current_degradation, "full_over_current": full_degradation,
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "rows": metric_rows,
        "no_further_exp009_tuning": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "targets": len(metric_rows), "components": components,
        "summaries": summaries, "bootstrap": bootstrap,
        "degradation": result["relative_absrel_degradation_fraction"],
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
