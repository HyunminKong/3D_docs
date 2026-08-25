#!/usr/bin/env python3
"""Diagnose metric/proxy alignment of the final atom without fitting a model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal, future_readout, track_objective
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _identifier
from revisit3d.scripts.evaluate_exp010_absolute_geometry import (
    LidarProjector,
    _depth_metrics,
    _query_lidar,
)


METRICS = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
ERROR_METRICS = ("silog", "abs_rel", "point_epe_m")


def _prediction_metrics(
    head: SpatialPlasticityHead,
    query: CachedAtomSegment,
    query_zero,
    context_atom,
    lidar: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    config: dict,
) -> dict | None:
    query_code = visual_transport(context_atom, query_zero).code
    depth = head.depth(query.features, query.base_depth, query_code)[0, :, :, 0]
    prediction = depth.detach().cpu().numpy().reshape(query.base_depth[0].shape)
    return _depth_metrics(
        prediction,
        lidar,
        valid,
        intrinsics,
        image_size=query.image_size,
        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
    )


def _summary(rows: list[dict], policy: str) -> dict:
    return {
        "targets": len(rows),
        **{metric: float(np.mean([row[policy][metric] for row in rows])) for metric in METRICS},
    }


def _component_bootstrap(
    rows: list[dict], left: str, right: str, metric: str, *, samples: int, seed: int
) -> dict:
    components = sorted({row["component"] for row in rows})
    sign = -1.0 if metric == "delta1" else 1.0
    values = np.asarray([
        sign * np.mean([
            row[left][metric] - row[right][metric]
            for row in rows if row["component"] == component
        ])
        for component in components
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": f"{left}_minus_{right}_positive_means_{right}_better",
        "components": len(components),
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _correlations(rows: list[dict]) -> dict:
    proxy = np.asarray([row["proxy_current_utility"] for row in rows], dtype=np.float64)
    output = {}
    for metric in ERROR_METRICS:
        total = np.asarray([
            row["base"][metric] - row["final_current"][metric] for row in rows
        ], dtype=np.float64)
        online = np.asarray([
            row["final_zero"][metric] - row["final_current"][metric] for row in rows
        ], dtype=np.float64)
        total_rho, total_p = spearmanr(proxy, total)
        online_rho, online_p = spearmanr(proxy, online)
        output[metric] = {
            "proxy_vs_base_to_current": {"rho": float(total_rho), "pvalue": float(total_p)},
            "proxy_vs_zero_to_current": {"rho": float(online_rho), "pvalue": float(online_p)},
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-022_metric_alignment_diagnosis_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-022 result already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-022 is a CUDA train-only diagnosis")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(
        config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True
    )
    reference = json.loads(Path(config["model"]["reference_result"]).read_text())
    checkpoint = torch.load(
        config["model"]["final_atom_checkpoint"], map_location="cpu", weights_only=False
    )
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 225
        and geometry.get("split") == "train"
        and reference["selected_variant"] == "track3d_only_eta0.0125"
        and checkpoint["experiment"] == "EXP-015"
        and checkpoint["step_size"] == float(config["model"]["step_size"])
        and checkpoint["validation_accessed"] is False
        and checkpoint["test_accessed"] is False
    ):
        raise RuntimeError("EXP-022 input contract failed")

    targets = {}
    for index, row in enumerate(manifest):
        key = _identifier(row["a_prime"])
        candidate = {
            "id": key,
            "index": index,
            "segment": row["a_prime"],
            "component": f"component-{int(row['component_id'])}",
            "location": row["location"],
        }
        if key in targets and targets[key]["segment"]["query_frames"] != candidate["segment"]["query_frames"]:
            raise RuntimeError("duplicate target changed query frames")
        targets.setdefault(key, candidate)
    if len(targets) != 218:
        raise RuntimeError(f"expected 218 targets, got {len(targets)}")

    reference_rows = {row["episode"]: row for row in reference["rows"]}
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
    rows = []
    with torch.enable_grad():
        for target_index, target in enumerate(targets.values()):
            cached = geometry["rows"][target["index"]]["segments"]
            current = CachedAtomSegment.from_cache(cached["a_prime_context"], "current", device)
            query = CachedAtomSegment.from_cache(cached["a_prime_query"], "query", device)
            zero = current.atom(head)
            query_zero = query.atom(head)
            current_code = adapt_minimal(
                head, current, zero.code, step_size=float(config["model"]["step_size"])
            )
            current_atom = replace(zero, code=current_code.detach())
            zero_query_loss = track_objective(head, query, query_zero.code)
            current_query_loss = future_readout(head, current_atom, query, query_zero)
            proxy_utility = float((
                (zero_query_loss - current_query_loss)
                / zero_query_loss.detach().abs().clamp_min(1e-6)
            ).detach())

            lidar, valid = _query_lidar(
                projector, scene_root, target["segment"], query.base_depth.shape[-1]
            )
            intrinsics = query.intrinsics[0].detach().cpu().numpy()
            base = _depth_metrics(
                query.base_depth[0].detach().cpu().numpy(),
                lidar,
                valid,
                intrinsics,
                image_size=query.image_size,
                minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
            )
            final_zero = _prediction_metrics(
                head, query, query_zero, zero, lidar, valid, intrinsics, config
            )
            final_current = _prediction_metrics(
                head, query, query_zero, current_atom, lidar, valid, intrinsics, config
            )
            episode = f"target-{target['id']}"
            old = reference_rows.get(episode)
            if base is not None and final_zero is not None and final_current is not None and old is not None:
                rows.append({
                    "episode": episode,
                    "component": target["component"],
                    "location": target["location"],
                    "base": base,
                    "reference_current": old["track3d_only_eta0.0125"],
                    "final_zero": final_zero,
                    "final_current": final_current,
                    "proxy_current_utility": proxy_utility,
                })
            if target_index == 0 or (target_index + 1) % 25 == 0 or target_index + 1 == len(targets):
                print(json.dumps({
                    "processed": target_index + 1,
                    "targets": len(targets),
                    "valid": len(rows),
                }), flush=True)

    summaries = {
        policy: _summary(rows, policy)
        for policy in ("base", "reference_current", "final_zero", "final_current")
    }
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    comparisons = {
        name: {
            metric: _component_bootstrap(
                rows, left, right, metric, samples=samples, seed=seed + 10 * index + metric_index
            )
            for metric_index, metric in enumerate(ERROR_METRICS)
        }
        for index, (name, left, right) in enumerate((
            ("reference_vs_base", "base", "reference_current"),
            ("final_zero_vs_base", "base", "final_zero"),
            ("final_online_vs_zero", "final_zero", "final_current"),
            ("final_current_vs_base", "base", "final_current"),
            ("final_vs_reference", "reference_current", "final_current"),
        ))
    }
    reproduced_base = reference["summaries"]["base"]
    base_difference = max(
        abs(summaries["base"][metric] - reproduced_base[metric]) for metric in METRICS
    )
    components = len({row["component"] for row in rows})
    checks = {
        "coverage": len(rows) >= int(config["success"]["minimum_targets"])
        and components >= int(config["success"]["minimum_components"]),
        "base_reproduction": base_difference
        <= float(config["success"]["maximum_base_reproduction_difference"]),
    }
    final_mean_health = all(
        summaries["final_current"][metric] < summaries["base"][metric]
        for metric in ERROR_METRICS
    )
    result = {
        "experiment": "EXP-022",
        "stage": "metric_alignment_diagnosis_train",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "config": str(config_path),
        "targets": len(rows),
        "components": components,
        "summaries": summaries,
        "comparisons": comparisons,
        "proxy_metric_correlations": _correlations(rows),
        "diagnosis": {
            "final_current_all_primary_means_improve": final_mean_health,
            "base_reproduction_max_abs_difference": base_difference,
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "query_lidar_evaluation_only": True,
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
        "components": components,
        "summaries": summaries,
        "comparisons": comparisons,
        "proxy_metric_correlations": result["proxy_metric_correlations"],
        "diagnosis": result["diagnosis"],
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
