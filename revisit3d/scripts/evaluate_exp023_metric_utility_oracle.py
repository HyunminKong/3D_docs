#!/usr/bin/env python3
"""No-fit oracle audit of one sparse metric-geometry utility label."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.experiments.exp012_minimal import adapt_minimal, future_readout
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp010_absolute_geometry import (
    LidarProjector,
    _depth_metrics,
    _query_lidar,
)
from revisit3d.scripts.train_exp012_utility_selected_atom import _episode_segments


METRICS = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")


def _prediction(head, query, query_zero, atom) -> np.ndarray:
    code = visual_transport(atom, query_zero).code
    depth = head.depth(query.features, query.base_depth, code)[0, :, :, 0]
    return depth.detach().cpu().numpy().reshape(query.base_depth[0].shape)


def _metric_risk(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_cells: int,
    minimum_depth: float,
) -> float | None:
    rows = []
    for view in range(prediction.shape[0]):
        mask = valid[view] & np.isfinite(prediction[view]) & (prediction[view] > minimum_depth)
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].astype(np.float64)
        gt = target[view][mask].astype(np.float64)
        scale = float(np.median(gt / pred))
        aligned = np.clip(pred * scale, minimum_depth, None)
        rows.append(float(np.mean(np.abs(np.log(aligned) - np.log(gt)))))
    return float(np.mean(rows)) if rows else None


def _metrics(prediction, target, valid, intrinsics, query, config):
    return _depth_metrics(
        prediction,
        target,
        valid,
        intrinsics,
        image_size=query.image_size,
        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
    )


def _summary(rows: list[dict], policy: str) -> dict:
    components = sorted({row["component"] for row in rows})
    return {
        "targets": len(rows),
        "components": len(components),
        "metric_risk": float(np.mean([
            np.mean([row[policy]["metric_risk"] for row in rows if row["component"] == component])
            for component in components
        ])),
        **{
            metric: float(np.mean([
                np.mean([row[policy]["metrics"][metric] for row in rows if row["component"] == component])
                for component in components
            ]))
            for metric in METRICS
        },
    }


def _bootstrap(rows, left, right, value, *, samples, seed):
    components = sorted({row["component"] for row in rows})
    if value == "metric_risk":
        getter = lambda row, policy: row[policy]["metric_risk"]
        sign = 1.0
    else:
        getter = lambda row, policy: row[policy]["metrics"][value]
        sign = -1.0 if value == "delta1" else 1.0
    component_values = np.asarray([
        sign * np.mean([
            getter(row, left) - getter(row, right)
            for row in rows if row["component"] == component
        ])
        for component in components
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        component_values, size=(samples, len(component_values)), replace=True
    ).mean(axis=1)
    return {
        "direction": f"{left}_minus_{right}_positive_means_{right}_better",
        "components": len(components),
        "mean_improvement": float(component_values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-023_metric_utility_oracle_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-023 result already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-023 is a CUDA train-only oracle audit")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(
        config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True
    )
    checkpoint = torch.load(
        config["model"]["checkpoint"], map_location="cpu", weights_only=False
    )
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train"
        and cache.get("protocol_revision") == "v1.5"
        and checkpoint["experiment"] == "EXP-015"
        and checkpoint["step_size"] == float(config["method"]["step_size"])
        and checkpoint["reuse_strength"] == float(config["method"]["reuse_strength"])
        and checkpoint["validation_accessed"] is False
        and checkpoint["test_accessed"] is False
    ):
        raise RuntimeError("EXP-023 frozen-input contract failed")

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
    scene_root = Path(config["data"]["scene_root"])
    all_indices = list(range(len(manifest)))
    rows = []
    with torch.enable_grad():
        for index, manifest_row in enumerate(manifest):
            sources, labels, current, query = _episode_segments(
                cache, manifest, index, all_indices, config, device
            )
            zero = current.atom(head)
            query_zero = query.atom(head)
            current_code = adapt_minimal(
                head, current, zero.code, step_size=float(config["method"]["step_size"])
            )
            current_atom = replace(zero, code=current_code.detach())
            current_proxy = float(future_readout(head, current_atom, query, query_zero).detach())
            lidar, valid = _query_lidar(
                projector, scene_root, manifest_row["a_prime"], query.base_depth.shape[-1]
            )
            intrinsics = query.intrinsics[0].detach().cpu().numpy()
            current_prediction = _prediction(head, query, query_zero, current_atom)
            current_metrics = _metrics(
                current_prediction, lidar, valid, intrinsics, query, config
            )
            current_risk = _metric_risk(
                current_prediction,
                lidar,
                valid,
                minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
                minimum_depth=float(config["metric_label"]["minimum_depth"]),
            )
            candidates = []
            for source, label in zip(sources, labels):
                source_zero = source.atom(head)
                source_code = adapt_minimal(
                    head,
                    source,
                    source_zero.code,
                    step_size=float(config["method"]["step_size"]),
                )
                source_atom = replace(source_zero, code=source_code.detach())
                transported = visual_transport(source_atom, zero).code
                candidate_code = (
                    current_code + float(config["method"]["reuse_strength"]) * transported
                ).clamp(-1, 1)
                candidate_atom = replace(zero, code=candidate_code.detach())
                proxy_loss = float(future_readout(head, candidate_atom, query, query_zero).detach())
                prediction = _prediction(head, query, query_zero, candidate_atom)
                metrics = _metrics(prediction, lidar, valid, intrinsics, query, config)
                risk = _metric_risk(
                    prediction,
                    lidar,
                    valid,
                    minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
                    minimum_depth=float(config["metric_label"]["minimum_depth"]),
                )
                candidates.append({
                    "label": label,
                    "proxy_loss": proxy_loss,
                    "metric_risk": risk,
                    "metrics": metrics,
                })
            if current_metrics is not None and current_risk is not None and all(
                candidate["metrics"] is not None and candidate["metric_risk"] is not None
                for candidate in candidates
            ):
                metric_oracle = min(candidates, key=lambda row: row["metric_risk"])
                proxy_oracle = min(candidates, key=lambda row: row["proxy_loss"])
                random_expectation = {
                    "metric_risk": float(np.mean([row["metric_risk"] for row in candidates])),
                    "metrics": {
                        metric: float(np.mean([row["metrics"][metric] for row in candidates]))
                        for metric in METRICS
                    },
                }
                rows.append({
                    "index": index,
                    "episode": manifest_row["episode_id"],
                    "component": f"component-{int(manifest_row['component_id'])}",
                    "location": manifest_row["location"],
                    "current": {"metric_risk": current_risk, "metrics": current_metrics},
                    "metric_oracle": metric_oracle,
                    "proxy_oracle": proxy_oracle,
                    "random_expectation": random_expectation,
                    "current_proxy_loss": current_proxy,
                    "candidate_metric_utilities": [
                        float((current_risk - row["metric_risk"]) / max(abs(current_risk), 1e-8))
                        for row in candidates
                    ],
                })
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(manifest):
                print(json.dumps({
                    "processed": index + 1,
                    "episodes": len(manifest),
                    "valid": len(rows),
                }), flush=True)

    policies = ("current", "metric_oracle", "proxy_oracle", "random_expectation")
    summaries = {policy: _summary(rows, policy) for policy in policies}
    primary = tuple(config["success"]["primary_metrics"])
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    comparisons = {
        "metric_oracle_vs_current": {
            value: _bootstrap(
                rows, "current", "metric_oracle", value,
                samples=samples, seed=seed + value_index,
            )
            for value_index, value in enumerate(("metric_risk", *primary))
        },
        "metric_oracle_vs_proxy_oracle": {
            value: _bootstrap(
                rows, "proxy_oracle", "metric_oracle", value,
                samples=samples, seed=seed + 10 + value_index,
            )
            for value_index, value in enumerate(("metric_risk", *primary))
        },
        "metric_oracle_vs_random": {
            value: _bootstrap(
                rows, "random_expectation", "metric_oracle", value,
                samples=samples, seed=seed + 20 + value_index,
            )
            for value_index, value in enumerate(("metric_risk", *primary))
        },
    }
    success = config["success"]
    checks = {
        "coverage": len(rows) >= int(success["minimum_targets"])
        and len({row["component"] for row in rows}) >= int(success["minimum_components"]),
        "all_metric_means_improve_vs_current": all(
            summaries["metric_oracle"][metric] < summaries["current"][metric]
            for metric in primary
        ),
        "positive_metric_intervals_vs_current": sum(
            comparisons["metric_oracle_vs_current"][metric]["ci95"][0] > 0
            for metric in primary
        ) >= int(success["minimum_positive_metric_intervals_vs_current"]),
        "all_metric_means_not_worse_than_proxy_oracle": all(
            summaries["metric_oracle"][metric] <= summaries["proxy_oracle"][metric]
            for metric in primary
        ),
        "metric_risk_interval_over_proxy_oracle":
            comparisons["metric_oracle_vs_proxy_oracle"]["metric_risk"]["ci95"][0] > 0,
    }
    metric_utilities = np.asarray([
        value for row in rows for value in row["candidate_metric_utilities"]
    ], dtype=np.float64)
    result = {
        "experiment": "EXP-023",
        "stage": "metric_utility_oracle_train",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "config": str(config_path),
        "targets": len(rows),
        "components": len({row["component"] for row in rows}),
        "metric_label": config["metric_label"]["name"],
        "summaries": summaries,
        "comparisons": comparisons,
        "candidate_metric_utility": {
            "mean": float(metric_utilities.mean()),
            "harmful_rate": float(np.mean(metric_utilities < 0)),
            "beneficial_rate": float(np.mean(metric_utilities > 0)),
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "no_model_fit": True,
        "query_lidar_evaluation_label_only": True,
        "query_or_future_online_input": False,
        "validation_accessed": False,
        "test_accessed": False,
        "exp021_terminal_accessed": False,
        "rows": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path),
        "targets": len(rows),
        "components": result["components"],
        "summaries": summaries,
        "comparisons": comparisons,
        "candidate_metric_utility": result["candidate_metric_utility"],
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
