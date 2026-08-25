#!/usr/bin/env python3
"""Train-only metric health sweep for minimal one-step TTT objectives."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.experiments import CachedAtomSegment, geometry_objective
from revisit3d.losses import track_3d_consistency_loss, track_reprojection_consistency_loss
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _identifier
from revisit3d.scripts.evaluate_exp010_absolute_geometry import LidarProjector, _depth_metrics, _query_lidar


def _single_loss(head, segment: CachedAtomSegment, code: torch.Tensor, objective: str) -> torch.Tensor:
    depth = head.depth(segment.features, segment.base_depth, code)
    side = int(depth.shape[2] ** 0.5)
    grid = depth.squeeze(-1).reshape(depth.shape[0], depth.shape[1], side, side)
    function = {
        "track3d_only": track_3d_consistency_loss,
        "track_reprojection": track_reprojection_consistency_loss,
    }[objective]
    return function(
        grid, segment.intrinsics, segment.predicted_w2c, segment.track,
        segment.track_visibility, segment.track_confidence, image_size=segment.image_size,
    )


def _adapt(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    initial: torch.Tensor,
    *,
    objective: str,
    step_size: float,
) -> torch.Tensor:
    if objective == "registered_full":
        loss = lambda _depth, code: geometry_objective(head, segment, code)
    else:
        loss = lambda _depth, code: _single_loss(head, segment, code, objective)
    code, _ = head.online_update(
        segment.features, segment.base_depth, initial, loss, step_size=step_size, steps=1,
    )
    return code


def _summary(rows: list[dict], variant: str) -> dict:
    metrics = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
    return {
        "targets": len(rows),
        **{metric: float(np.mean([row[variant][metric] for row in rows])) for metric in metrics},
    }


def _bootstrap(
    rows: list[dict], variant: str, metric: str, *, samples: int, seed: int,
) -> dict:
    by_component: dict[str, list[float]] = {}
    for row in rows:
        improvement = row["base"][metric] - row[variant][metric]
        by_component.setdefault(row["component"], []).append(float(improvement))
    values = np.asarray([np.mean(by_component[key]) for key in sorted(by_component)], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": "base_error_minus_adapted_error", "components": len(values),
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-011_objective_health_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-011 result already exists")
    if config["data"]["split"] != "train":
        raise RuntimeError("EXP-011 objective selection must remain train-only")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-011 requires CUDA")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train"
        and cache.get("protocol_revision") == "v1.5"
    ):
        raise RuntimeError("EXP-011 train cache contract failed")
    checkpoint = torch.load(config["model"]["checkpoint"], map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("split") == "train"
        and checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("EXP-011 requires the frozen train plasticity head")
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
    variants = [
        (objective, float(step_size), f"{objective}_eta{float(step_size):g}")
        for objective in config["variants"]["objectives"]
        for step_size in config["variants"]["step_sizes"]
    ]
    targets = {}
    for index, row in enumerate(manifest):
        key = _identifier(row["a_prime"])
        candidate = {
            "id": key, "index": index, "segment": row["a_prime"],
            "component": f"component-{int(row['component_id'])}", "location": row["location"],
        }
        if key in targets and targets[key]["segment"]["query_frames"] != candidate["segment"]["query_frames"]:
            raise RuntimeError("duplicate train target has inconsistent query frames")
        targets.setdefault(key, candidate)
    if len(targets) != 218:
        raise RuntimeError(f"expected 218 unique train targets, got {len(targets)}")
    scene_root = Path(config["data"]["scene_root"])
    rows = []
    with torch.enable_grad():
        for target_index, target in enumerate(targets.values()):
            cached = cache["rows"][target["index"]]["segments"]
            current = CachedAtomSegment.from_cache(cached["a_prime_context"], "current", device)
            query = CachedAtomSegment.from_cache(cached["a_prime_query"], "query", device)
            zero = current.atom(head)
            query_zero = query.atom(head)
            base_depth = query.base_depth[0].detach().cpu().numpy()
            side = base_depth.shape[-1]
            lidar_depth, lidar_valid = _query_lidar(projector, scene_root, target["segment"], side)
            intrinsics = query.intrinsics[0].detach().cpu().numpy()
            base_metrics = _depth_metrics(
                base_depth, lidar_depth, lidar_valid, intrinsics,
                image_size=query.image_size,
                minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
            )
            row = {
                "episode": f"target-{target['id']}", "component": target["component"],
                "location": target["location"], "base": base_metrics,
            }
            for objective, step_size, label in variants:
                code = _adapt(head, current, zero.code, objective=objective, step_size=step_size)
                query_code = visual_transport(replace(zero, code=code), query_zero).code
                prediction = head.depth(query.features, query.base_depth, query_code)[0, :, :, 0]
                prediction = prediction.detach().cpu().numpy().reshape(base_depth.shape)
                row[label] = _depth_metrics(
                    prediction, lidar_depth, lidar_valid, intrinsics,
                    image_size=query.image_size,
                    minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
                )
            if base_metrics is not None and all(row[label] is not None for _, _, label in variants):
                rows.append(row)
            if target_index == 0 or (target_index + 1) % 25 == 0 or target_index + 1 == len(targets):
                print(json.dumps({"processed": target_index + 1, "targets": len(targets), "valid": len(rows)}), flush=True)

    summaries = {"base": _summary(rows, "base")}
    bootstrap, health = {}, {}
    primary = tuple(config["success"]["primary_metrics"])
    base = summaries["base"]
    for variant_index, (_, step_size, label) in enumerate(variants):
        summaries[label] = _summary(rows, label)
        bootstrap[label] = {
            metric: _bootstrap(
                rows, label, metric, samples=int(config["statistics"]["bootstrap_samples"]),
                seed=int(config["statistics"]["bootstrap_seed"]) + 10 * variant_index + metric_index,
            ) for metric_index, metric in enumerate(primary)
        }
        relative = {metric: (base[metric] - summaries[label][metric]) / base[metric] for metric in primary}
        checks = {
            "all_primary_means_improve": all(relative[metric] > 0 for metric in primary),
            "positive_intervals": sum(bootstrap[label][metric]["ci95"][0] > 0 for metric in primary)
            >= int(config["success"]["minimum_positive_intervals"]),
        }
        health[label] = {
            "step_size": step_size, "relative_improvement": relative,
            "worst_relative_improvement": min(relative.values()),
            "checks": checks, "passed": all(checks.values()),
        }
    eligible = [label for _, _, label in variants if health[label]["passed"]]
    selected = max(
        eligible,
        key=lambda label: (
            health[label]["worst_relative_improvement"],
            label.startswith("track3d_only") or label.startswith("track_reprojection"),
            -health[label]["step_size"],
        ),
    ) if eligible else None
    components = len({row["component"] for row in rows})
    coverage = (
        len(rows) >= int(config["success"]["minimum_targets"])
        and components >= int(config["success"]["minimum_components"])
    )
    result = {
        "experiment": "EXP-011", "stage": "stage0_objective_health_train",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "query_lidar_evaluation_only": True, "query_or_future_online_input": False,
        "config": str(config_path), "targets": len(rows), "components": components,
        "variants": [label for _, _, label in variants], "summaries": summaries,
        "bootstrap": bootstrap, "health": health,
        "selected_variant": selected,
        "registered_gate": {"coverage": coverage, "passed": coverage and selected is not None},
        "validation_accessed": False, "test_accessed": False,
        "rows": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "targets": len(rows), "components": components,
        "summaries": summaries, "health": health, "selected": selected,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
