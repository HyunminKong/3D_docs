#!/usr/bin/env python3
"""One-shot proxy and sparse-LiDAR validation of the frozen paper model."""

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

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal, future_readout, track_objective
from revisit3d.models import PlasticityAtom, SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _cpu_atom, _device_atom, _identifier, _timestamp
from revisit3d.scripts.evaluate_exp010_absolute_geometry import LidarProjector, _depth_metrics, _query_lidar
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _inventory(manifest: list[dict]):
    context, targets = {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context")):
            segment = row[tag]
            key = _identifier(segment)
            info = {
                "id": key, "segment": segment, "cache_index": index, "cache_tag": cache_tag,
                "location": row["location"],
            }
            if key in context and context[key]["segment"] != segment:
                raise RuntimeError("validation duplicate context changed")
            context.setdefault(key, info)
        target_key = _identifier(row["a_prime"])
        target = {
            "id": target_key, "segment": row["a_prime"], "cache_index": index,
            "episode": f"target-{target_key}", "component": f"component-{int(row['component_id'])}",
            "location": row["location"],
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        }
        if target_key in targets and targets[target_key]["query_frames"] != target["query_frames"]:
            raise RuntimeError("validation duplicate query changed")
        targets.setdefault(target_key, target)
    if len(targets) != 103 or len(set(row["component"] for row in targets.values())) != 17:
        raise RuntimeError("validation target/component inventory changed")
    return context, targets


def _pair(current: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.concatenate((current, source, current - source, current * source))


def _query_metrics(head, query, query_zero, context_atom, lidar, valid, intrinsics, config):
    code = visual_transport(context_atom, query_zero).code
    depth = head.depth(query.features, query.base_depth, code)[0, :, :, 0]
    base_shape = query.base_depth[0].shape
    prediction = depth.detach().cpu().numpy().reshape(base_shape)
    return _depth_metrics(
        prediction, lidar, valid, intrinsics, image_size=query.image_size,
        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
    )


def _average_metrics(rows: list[dict]) -> dict:
    keys = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
    return {
        "valid_views": float(np.mean([row["valid_views"] for row in rows])),
        "valid_cells": float(np.mean([row["valid_cells"] for row in rows])),
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
    }


def _summary(rows: list[dict], policy: str) -> dict:
    metrics = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")
    components = sorted({row["component"] for row in rows})
    return {
        "targets": len(rows), "components": len(components),
        **{
            metric: float(np.mean([
                np.mean([row[policy][metric] for row in rows if row["component"] == component])
                for component in components
            ])) for metric in metrics
        },
    }


def _bootstrap_metric(rows, left, right, metric, *, samples, seed):
    components = sorted({row["component"] for row in rows})
    # Positive means the right policy has lower error than the left policy.
    values = np.asarray([
        np.mean([
            row[left][metric] - row[right][metric]
            for row in rows if row["component"] == component
        ]) for component in components
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": f"{left}_error_minus_{right}_error", "components": len(values),
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _bootstrap_proxy(rows, *, samples, seed):
    components = sorted({row["component"] for row in rows})
    values = np.asarray([
        np.mean([
            row["proxy_full_utility"] - row["proxy_random_utility"]
            for row in rows if row["component"] == component
        ]) for component in components
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": "full_minus_matched_acceptance_random_proxy_utility",
        "mean_difference": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-020_paper_model_validation_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-020 locked validation result already exists")
    if config["data"]["split"] != "val" or not torch.cuda.is_available():
        raise RuntimeError("EXP-020 requires validation split and CUDA")
    atom_result = json.loads(Path(config["model"]["atom_result"]).read_text())
    address_result = json.loads(Path(config["model"]["address_result"]).read_text())
    atom_path = Path(config["model"]["atom_checkpoint"])
    artifact_path = Path(config["model"]["address_artifact"])
    if not (
        atom_result["gate"]["passed"] is True and atom_result["checkpoint_sha256"] == _sha256(atom_path)
        and address_result["registered_gate"]["passed"] is True
        and address_result["artifact_sha256"] == _sha256(artifact_path)
        and address_result["validation_accessed"] is False and address_result["test_accessed"] is False
    ):
        raise RuntimeError("frozen paper artifact contract failed")
    checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    artifact = joblib.load(artifact_path)
    if not (
        artifact["experiment"] == "EXP-019" and artifact["topk"] == config["method"]["coarse_topk"]
        and artifact["utility_threshold"] == config["method"]["utility_threshold"] == 0.0
        and artifact["geometry_agreement_threshold"] == config["method"]["geometry_agreement_threshold"] == 0.0
        and artifact["validation_accessed"] is False and artifact["test_accessed"] is False
        and checkpoint["step_size"] == config["method"]["step_size"]
        and checkpoint["reuse_strength"] == config["method"]["reuse_strength"]
    ):
        raise RuntimeError("paper method fields changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 117
        and geometry.get("split") == "val" and geometry.get("protocol_revision") == "v2.2"
    ):
        raise RuntimeError("validation cache contract failed")
    context, targets = _inventory(manifest)
    metadata_cache = {}
    scene_root = Path(config["data"]["scene_root"])
    by_location = {}
    for info in context.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)
        by_location.setdefault(info["location"], []).append(info)
    for events in by_location.values():
        events.sort(key=lambda row: (row["timestamp"], row["id"]))
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
    model = artifact["coarse_model"]
    capacity = int(config["method"]["bank_capacity"])
    rows = []
    with torch.enable_grad():
        for location, events in sorted(by_location.items()):
            stable = int(hashlib.sha1(location.encode()).hexdigest()[:8], 16)
            generator = random.Random(int(config["seed"]) + stable)
            bank: list[dict] = []
            seen = 0
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code = adapt_minimal(
                    head, segment, zero.code, step_size=float(config["method"]["step_size"]),
                )
                atom = replace(zero, code=code.detach())
                descriptor = zero.key.mean(dim=(1, 2))[0].detach().cpu().numpy().astype(np.float64)
                if key in targets:
                    target = targets[key]
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    query_zero = query.atom(head)
                    current_query_loss = future_readout(head, atom, query, query_zero)
                    lidar, lidar_valid = _query_lidar(projector, scene_root, target["segment"], query.base_depth.shape[-1])
                    intrinsics = query.intrinsics[0].detach().cpu().numpy()
                    base_depth = query.base_depth[0].detach().cpu().numpy()
                    base_metrics = _depth_metrics(
                        base_depth, lidar, lidar_valid, intrinsics, image_size=query.image_size,
                        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
                    )
                    current_metrics = _query_metrics(
                        head, query, query_zero, atom, lidar, lidar_valid, intrinsics, config,
                    )
                    candidate_rows = []
                    current_online = track_objective(head, segment, code)
                    for source_state in bank:
                        source = _device_atom(source_state["atom"], device)
                        transported = visual_transport(source, zero).code
                        candidate_code = (
                            code + float(config["method"]["reuse_strength"]) * transported
                        ).clamp(-1, 1)
                        candidate_atom = replace(zero, code=candidate_code)
                        candidate_online = track_objective(head, segment, candidate_code)
                        agreement = float((
                            (current_online - candidate_online)
                            / current_online.detach().abs().clamp_min(1e-6)
                        ).detach())
                        score = float(model.predict(_pair(descriptor, source_state["descriptor"])[None])[0])
                        candidate_query = future_readout(head, candidate_atom, query, query_zero)
                        utility = float((
                            (current_query_loss - candidate_query)
                            / current_query_loss.detach().abs().clamp_min(1e-6)
                        ).detach())
                        metrics = _query_metrics(
                            head, query, query_zero, candidate_atom,
                            lidar, lidar_valid, intrinsics, config,
                        )
                        appearance = float(
                            descriptor @ source_state["descriptor"]
                            / max(np.linalg.norm(descriptor) * np.linalg.norm(source_state["descriptor"]), 1e-12)
                        )
                        candidate_rows.append({
                            "id": source_state["id"], "score": score, "agreement": agreement,
                            "utility": utility, "metrics": metrics, "appearance": appearance,
                        })
                    full_metrics = coarse_metrics = appearance_metrics = random_metrics = current_metrics
                    full_utility = random_utility = 0.0
                    accepted = False
                    if candidate_rows:
                        ranked = sorted(candidate_rows, key=lambda row: (-row["score"], row["id"]))
                        coarse = ranked[0]
                        topk = ranked[: min(int(config["method"]["coarse_topk"]), len(ranked))]
                        safe = [row for row in topk if row["score"] > 0 and row["agreement"] > 0]
                        winner = max(safe, key=lambda row: (row["agreement"], row["id"])) if safe else coarse
                        accepted = winner["score"] > 0
                        if accepted:
                            full_metrics, full_utility = winner["metrics"], winner["utility"]
                            random_metrics = _average_metrics([row["metrics"] for row in candidate_rows])
                            random_utility = float(np.mean([row["utility"] for row in candidate_rows]))
                            appearance_metrics = max(candidate_rows, key=lambda row: row["appearance"])["metrics"]
                        if coarse["score"] > 0:
                            coarse_metrics = coarse["metrics"]
                    if base_metrics is not None and all(value is not None for value in (
                        current_metrics, full_metrics, random_metrics, coarse_metrics, appearance_metrics,
                    )):
                        rows.append({
                            "episode": target["episode"], "component": target["component"],
                            "location": location, "bank_size": len(bank), "accepted": accepted,
                            "base": base_metrics, "current": current_metrics, "full": full_metrics,
                            "random": random_metrics, "coarse": coarse_metrics, "appearance": appearance_metrics,
                            "proxy_full_utility": full_utility, "proxy_random_utility": random_utility,
                        })
                state = {
                    "id": key, "atom": _cpu_atom(atom), "descriptor": descriptor,
                }
                seen += 1
                if len(bank) < capacity:
                    bank.append(state)
                else:
                    replacement = generator.randrange(seen)
                    if replacement < capacity:
                        bank[replacement] = state
            print(json.dumps({
                "location": location, "events": len(events),
                "valid_targets": sum(row["location"] == location for row in rows),
            }), flush=True)
    summaries = {policy: _summary(rows, policy) for policy in (
        "base", "current", "full", "random", "coarse", "appearance",
    )}
    primary = tuple(config["success"]["primary_metrics"])
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    bootstrap = {
        "full_vs_current": {
            metric: _bootstrap_metric(rows, "current", "full", metric, samples=samples, seed=seed + index)
            for index, metric in enumerate(primary)
        },
        "full_vs_random": {
            metric: _bootstrap_metric(rows, "random", "full", metric, samples=samples, seed=seed + 10 + index)
            for index, metric in enumerate(primary)
        },
        "proxy_full_vs_random": _bootstrap_proxy(rows, samples=samples, seed=seed + 20),
    }
    proxy = {
        "full_mean_utility": float(np.mean([row["proxy_full_utility"] for row in rows])),
        "random_mean_utility": float(np.mean([row["proxy_random_utility"] for row in rows])),
        "harmful_rate": float(np.mean([row["proxy_full_utility"] < 0 for row in rows])),
        "acceptance": float(np.mean([row["accepted"] for row in rows])),
    }
    components = len({row["component"] for row in rows})
    checks = {
        "coverage": len(rows) >= int(config["success"]["minimum_targets"]) and components >= int(config["success"]["minimum_components"]),
        "current_all_mean_improve": all(summaries["current"][metric] < summaries["base"][metric] for metric in primary),
        "full_not_worse_current_all_means": all(summaries["full"][metric] <= summaries["current"][metric] for metric in primary),
        "full_not_worse_random_all_means": all(summaries["full"][metric] <= summaries["random"][metric] for metric in primary),
        "positive_full_current_intervals": sum(
            bootstrap["full_vs_current"][metric]["ci95"][0] > 0 for metric in primary
        ) >= int(config["success"]["minimum_positive_full_current_intervals"]),
        "positive_full_random_intervals": sum(
            bootstrap["full_vs_random"][metric]["ci95"][0] > 0 for metric in primary
        ) >= int(config["success"]["minimum_positive_full_random_intervals"]),
        "proxy_random_interval_positive": bootstrap["proxy_full_vs_random"]["ci95"][0] > 0,
        "proxy_harm": proxy["harmful_rate"] <= float(config["success"]["maximum_proxy_harmful_rate"]),
        "minimum_acceptance": proxy["acceptance"] >= float(config["success"]["minimum_acceptance"]),
    }
    result = {
        "experiment": "EXP-020", "stage": "locked_paper_model_validation", "split": "val",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "targets": len(rows), "components": components, "summaries": summaries,
        "proxy": proxy, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "query_lidar_evaluation_only": True, "query_or_future_online_input": False,
        "test_accessed": False, "rows": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "targets": len(rows), "components": components,
        "summaries": summaries, "proxy": proxy, "bootstrap": bootstrap,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
