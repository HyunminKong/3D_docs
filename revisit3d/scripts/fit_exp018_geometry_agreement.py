#!/usr/bin/env python3
"""Coarse utility MIPS plus deterministic current-geometry agreement."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal, track_objective
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _cpu_atom, _device_atom
from revisit3d.scripts.fit_exp016_unified_utility_address import (
    _bootstrap,
    _compile,
    _component_summary,
    _context_tables,
    _sha256,
    _strict_oof,
)


def _context_atoms(head, geometry, manifest, *, step_size: float, device):
    context, targets, _ = _context_tables(manifest)
    states = {}
    with torch.enable_grad():
        for count, info in enumerate(context.values(), start=1):
            payload = geometry["rows"][info["cache_index"]]["segments"][info["cache_tag"]]
            role = "current" if info["id"] in targets else "source"
            segment = CachedAtomSegment.from_cache(payload, role, device)
            zero = segment.atom(head)
            code = adapt_minimal(head, segment, zero.code, step_size=step_size)
            states[info["id"]] = _cpu_atom(replace(zero, code=code.detach()))
            if count % 100 == 0 or count == len(context):
                print(json.dumps({"context_atoms": count, "total": len(context)}), flush=True)
    return states, context, targets


def _agreement(head, geometry, metadata, states, context, *, strength: float, device):
    by_target = {}
    for index, row in enumerate(metadata):
        by_target.setdefault(row["target_context"], []).append(index)
    values = torch.empty(len(metadata), dtype=torch.float32)
    with torch.enable_grad():
        for count, (target_key, indices) in enumerate(by_target.items(), start=1):
            info = context[target_key]
            payload = geometry["rows"][info["cache_index"]]["segments"][info["cache_tag"]]
            current = CachedAtomSegment.from_cache(payload, "current", device)
            current_atom = _device_atom(states[target_key], device)
            current_zero = replace(current_atom, code=torch.zeros_like(current_atom.code))
            current_loss = track_objective(head, current, current_atom.code)
            for index in indices:
                source = _device_atom(states[metadata[index]["source_context"]], device)
                transported = visual_transport(source, current_zero).code
                candidate_code = (current_atom.code + strength * transported).clamp(-1, 1)
                candidate_loss = track_objective(head, current, candidate_code)
                values[index] = float(
                    ((current_loss - candidate_loss) / current_loss.detach().abs().clamp_min(1e-6)).detach()
                )
            if count % 25 == 0 or count == len(by_target):
                print(json.dumps({"agreement_targets": count, "total": len(by_target)}), flush=True)
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite current geometry agreement")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-018_geometry_agreement_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    agreement_path = Path(config["output"]["agreement_cache"])
    artifact_path = Path(config["output"]["artifact"])
    if result_path.exists() or agreement_path.exists() or artifact_path.exists():
        raise RuntimeError("EXP-018 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-018 requires train split and CUDA")
    visual_result = json.loads(Path(config["prior_visual_result"]).read_text())
    adaptation_result = json.loads(Path(config["prior_adaptation_result"]).read_text())
    if not (
        visual_result["registered_gate"]["passed"] is False
        and adaptation_result["registered_gate"]["passed"] is False
        and visual_result["registered_gate"]["checks"]["random_component_ci_positive"] is False
        and adaptation_result["registered_gate"]["checks"]["random_component_ci_positive"] is False
        and visual_result["validation_accessed"] is False and adaptation_result["validation_accessed"] is False
    ):
        raise RuntimeError("EXP-016/017 failure contract changed")
    pair_path = Path(config["data"]["candidate_cache"])
    if visual_result["candidate_cache_sha256"] != _sha256(pair_path):
        raise RuntimeError("visual candidate cache hash changed")
    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    atom_result = json.loads(Path(config["model"]["atom_result"]).read_text())
    checkpoint_path = Path(config["model"]["atom_checkpoint"])
    if atom_result["checkpoint_sha256"] != _sha256(checkpoint_path) or atom_result["gate"]["passed"] is not True:
        raise RuntimeError("frozen atom contract failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["model"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    states, context, _ = _context_atoms(
        head, geometry, manifest, step_size=float(config["method"]["step_size"]), device=device,
    )
    agreement = _agreement(
        head, geometry, pair["metadata"], states, context,
        strength=float(config["method"]["reuse_strength"]), device=device,
    )
    agreement_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-018", "protocol_revision": config["protocol_revision"], "split": "train",
        "current_geometry_agreement": agreement, "metadata": pair["metadata"],
        "candidate_cache_sha256": _sha256(pair_path), "validation_accessed": False, "test_accessed": False,
    }, agreement_path)
    matrix = pair["features"].numpy().astype(np.float64)
    utility = pair["utility"].numpy().astype(np.float64)
    metadata, target_table = pair["metadata"], pair["target_table"]
    target_location = np.asarray([row["target_location"] for row in metadata])
    source_location = np.asarray([row["source_location"] for row in metadata])
    prediction, folds = _strict_oof(
        matrix, utility, target_location, source_location, float(config["address"]["ridge_alpha"]),
    )
    by_episode = {}
    for index, row in enumerate(metadata):
        by_episode.setdefault(row["episode"], []).append(index)
    full, coarse, random_same_accept, accepted = {}, {}, {}, {}
    k = int(config["method"]["coarse_topk"])
    for episode in target_table:
        indices = by_episode.get(episode, [])
        if not indices:
            full[episode] = coarse[episode] = random_same_accept[episode] = 0.0
            accepted[episode] = False
            continue
        ranked = sorted(indices, key=lambda index: (-prediction[index], metadata[index]["source_context"]))
        topk = ranked[: min(k, len(ranked))]
        winner = max(topk, key=lambda index: (float(agreement[index]), metadata[index]["source_context"]))
        take = bool(
            prediction[winner] > float(config["method"]["utility_threshold"])
            and agreement[winner] > float(config["method"]["geometry_agreement_threshold"])
        )
        accepted[episode] = take
        full[episode] = float(utility[winner]) if take else 0.0
        coarse_winner = ranked[0]
        coarse[episode] = float(utility[coarse_winner]) if prediction[coarse_winner] > 0 else 0.0
        random_same_accept[episode] = float(np.mean(utility[indices])) if take else 0.0
    policies = {
        "geometry_agreement": _component_summary(full, target_table, accepted),
        "coarse_top1": _component_summary(coarse, target_table, {key: value != 0 for key, value in coarse.items()}),
        "random_same_accept": _component_summary(random_same_accept, target_table, accepted),
    }
    bootstrap = {
        "full_minus_random": _bootstrap(
            full, random_same_accept, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "full_minus_coarse": _bootstrap(
            full, coarse, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    primary = policies["geometry_agreement"]
    success = config["success"]
    checks = {
        "minimum_policy_utility": primary["mean_utility"] > float(success["minimum_policy_utility"]),
        "minimum_acceptance": primary["acceptance"] >= float(success["minimum_acceptance"]),
        "maximum_harm": primary["harmful_rate"] <= float(success["maximum_harmful_rate"]),
        "random_component_ci_positive": bootstrap["full_minus_random"]["ci95"][0] > 0,
        "coarse_component_ci_positive": bootstrap["full_minus_coarse"]["ci95"][0] > 0,
    }
    result = {
        "experiment": "EXP-018", "stage": "geometry_agreement", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "targets": len(target_table), "pairs": len(metadata), "components": 25,
        "coarse_topk": k, "learned_fine_router": False,
        "future_or_query_online_input": False, "folds": folds,
        "agreement_cache": str(agreement_path), "agreement_cache_sha256": _sha256(agreement_path),
        "policies": policies, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "validation_accessed": False, "test_accessed": False,
    }
    if all(checks.values()):
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["address"]["ridge_alpha"])))
        model.fit(matrix, utility)
        compiled = _compile(model, matrix)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "experiment": "EXP-018", "protocol_revision": config["protocol_revision"],
            "split": "train", "coarse_model": model, "compiled_mips": compiled,
            "descriptor_dim": 64, "topk": k, "utility_threshold": 0.0,
            "fine_rule": "argmax_current_geometry_agreement_then_positive_zero_gate",
            "geometry_agreement_threshold": 0.0, "atom_checkpoint": str(checkpoint_path),
            "atom_checkpoint_sha256": _sha256(checkpoint_path),
            "validation_accessed": False, "test_accessed": False,
        }, artifact_path)
        result["artifact"] = str(artifact_path)
        result["artifact_sha256"] = _sha256(artifact_path)
        result["mips_maximum_verification_error"] = compiled["maximum_verification_error"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "policies": policies, "bootstrap": bootstrap,
        "gate": result["registered_gate"], "artifact": result.get("artifact"),
    }), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"EXP-018 geometry-agreement gate failed: {checks}")


if __name__ == "__main__":
    main()
