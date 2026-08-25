#!/usr/bin/env python3
"""One-shot validation of the train-selected minimal TTT objective."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.experiments import CachedAtomSegment
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _identifier
from revisit3d.scripts.evaluate_exp010_absolute_geometry import LidarProjector, _depth_metrics, _query_lidar
from revisit3d.scripts.evaluate_exp011_objective_health import _adapt, _bootstrap, _summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-011_locked_validation_v11.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-011 locked validation result already exists")
    if config["data"]["split"] != "val":
        raise RuntimeError("this evaluator is validation-only")
    if config["locked_objective"] != {
        "name": "track3d_only",
        "step_size": 0.0125,
        "steps": 1,
        "selected_by": "revisit3d/results/EXP-011/stage0_objective_health_train_v10.json",
    }:
        raise RuntimeError("locked Stage-0 objective contract changed")
    train_result = json.loads(Path(config["locked_objective"]["selected_by"]).read_text())
    if not (
        train_result["split"] == "train"
        and train_result["registered_gate"]["passed"] is True
        and train_result["selected_variant"] == "track3d_only_eta0.0125"
        and train_result["validation_accessed"] is False
        and train_result["test_accessed"] is False
    ):
        raise RuntimeError("Stage-0 selection artifact contract failed")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-011 validation requires CUDA")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(cache.get("rows", [])) == 117
        and cache.get("split") == "val"
        and cache.get("protocol_revision") == "v2.2"
    ):
        raise RuntimeError("validation cache contract failed")
    checkpoint = torch.load(config["model"]["checkpoint"], map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("split") == "train"
        and checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("frozen train plasticity-head contract failed")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["model"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    projector = LidarProjector(
        config["data"]["nuscenes_root"],
        minimum_depth=float(config["lidar"]["minimum_depth_m"]),
        maximum_depth=float(config["lidar"]["maximum_depth_m"]),
    )

    targets = {}
    for index, manifest_row in enumerate(manifest):
        key = _identifier(manifest_row["a_prime"])
        candidate = {
            "id": key, "index": index, "segment": manifest_row["a_prime"],
            "component": f"component-{int(manifest_row['component_id'])}",
            "location": manifest_row["location"],
        }
        if key in targets and targets[key]["segment"]["query_frames"] != candidate["segment"]["query_frames"]:
            raise RuntimeError("duplicate validation target has inconsistent query frames")
        targets.setdefault(key, candidate)
    if len(targets) != 103:
        raise RuntimeError(f"expected 103 unique validation targets, got {len(targets)}")

    scene_root = Path(config["data"]["scene_root"])
    rows = []
    with torch.enable_grad():
        for target_index, target in enumerate(targets.values()):
            cached = cache["rows"][target["index"]]["segments"]
            current = CachedAtomSegment.from_cache(cached["a_prime_context"], "current", device)
            query = CachedAtomSegment.from_cache(cached["a_prime_query"], "query", device)
            current_zero = current.atom(head)
            query_zero = query.atom(head)
            code = _adapt(
                head, current, current_zero.code, objective="track3d_only",
                step_size=float(config["locked_objective"]["step_size"]),
            )
            query_code = visual_transport(replace(current_zero, code=code), query_zero).code
            prediction = head.depth(query.features, query.base_depth, query_code)[0, :, :, 0]
            base_depth = query.base_depth[0].detach().cpu().numpy()
            prediction = prediction.detach().cpu().numpy().reshape(base_depth.shape)
            side = base_depth.shape[-1]
            lidar_depth, lidar_valid = _query_lidar(projector, scene_root, target["segment"], side)
            intrinsics = query.intrinsics[0].detach().cpu().numpy()
            metric_args = {
                "image_size": query.image_size,
                "minimum_cells": int(config["lidar"]["minimum_cells_per_view"]),
            }
            base_metrics = _depth_metrics(base_depth, lidar_depth, lidar_valid, intrinsics, **metric_args)
            current_metrics = _depth_metrics(prediction, lidar_depth, lidar_valid, intrinsics, **metric_args)
            if base_metrics is not None and current_metrics is not None:
                rows.append({
                    "episode": f"target-{target['id']}", "component": target["component"],
                    "location": target["location"], "base": base_metrics, "current": current_metrics,
                })
            if target_index == 0 or (target_index + 1) % 25 == 0 or target_index + 1 == len(targets):
                print(json.dumps({"processed": target_index + 1, "targets": len(targets), "valid": len(rows)}), flush=True)

    summaries = {"base": _summary(rows, "base"), "current": _summary(rows, "current")}
    primary = tuple(config["success"]["primary_metrics"])
    bootstrap = {
        metric: _bootstrap(
            rows, "current", metric,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + index,
        ) for index, metric in enumerate(primary)
    }
    improvements = {
        metric: (summaries["base"][metric] - summaries["current"][metric]) / summaries["base"][metric]
        for metric in primary
    }
    components = len({row["component"] for row in rows})
    checks = {
        "coverage": (
            len(rows) >= int(config["success"]["minimum_targets"])
            and components >= int(config["success"]["minimum_components"])
        ),
        "all_primary_means_improve": all(value > 0 for value in improvements.values()),
        "positive_intervals": sum(bootstrap[metric]["ci95"][0] > 0 for metric in primary)
        >= int(config["success"]["minimum_positive_intervals"]),
    }
    result = {
        "experiment": "EXP-011", "stage": "stage1_locked_objective_validation",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "query_lidar_evaluation_only": True, "query_or_future_online_input": False,
        "config": str(config_path), "targets": len(rows), "components": components,
        "locked_objective": config["locked_objective"], "summaries": summaries,
        "relative_improvement": improvements, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "test_accessed": False, "rows": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "targets": len(rows), "components": components,
        "summaries": summaries, "relative_improvement": improvements,
        "bootstrap": bootstrap, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
