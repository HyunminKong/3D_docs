#!/usr/bin/env python3
"""Add one observable adaptation-history scalar to the unified address."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal, track_objective
from revisit3d.models import SpatialPlasticityHead
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _identifier
from revisit3d.scripts.fit_exp016_unified_utility_address import (
    _bootstrap,
    _component_summary,
    _context_tables,
    _sha256,
    _strict_oof,
)


def _adaptation_context(head, geometry, manifest, *, step_size: float, device) -> dict[str, float]:
    context, _, _ = _context_tables(manifest)
    values = {}
    with torch.enable_grad():
        for count, info in enumerate(context.values(), start=1):
            payload = geometry["rows"][info["cache_index"]]["segments"][info["cache_tag"]]
            segment = CachedAtomSegment.from_cache(payload, "source", device)
            zero = segment.atom(head)
            pre = track_objective(head, segment, zero.code)
            code = adapt_minimal(head, segment, zero.code, step_size=step_size)
            post = track_objective(head, segment, code)
            values[info["id"]] = float(((pre - post) / pre.detach().abs().clamp_min(1e-6)).detach())
            if count % 100 == 0 or count == len(context):
                print(json.dumps({"context_states": count, "total": len(context)}), flush=True)
    if len(values) != 557 or not np.isfinite(list(values.values())).all():
        raise RuntimeError("adaptation-context state contract failed")
    return values


def _augment(prior: dict, context: dict[str, float]):
    old = prior["features"].float()
    metadata = prior["metadata"]
    if old.shape != (len(metadata), 256):
        raise RuntimeError("EXP-016 pair cache shape changed")
    current_visual, source_visual = old[:, :64], old[:, 64:128]
    current_history = torch.tensor([
        context[row["target_context"]] for row in metadata
    ], dtype=torch.float32).unsqueeze(1)
    source_history = torch.tensor([
        context[row["source_context"]] for row in metadata
    ], dtype=torch.float32).unsqueeze(1)
    current = torch.cat((current_visual, current_history), dim=1)
    source = torch.cat((source_visual, source_history), dim=1)
    matrix = torch.cat((current, source, current - source, current * source), dim=1)
    if matrix.shape != (len(metadata), 260) or not torch.isfinite(matrix).all():
        raise RuntimeError("65-D adaptation-context pair contract failed")
    return matrix, prior["utility"].float(), metadata, prior["target_table"]


def _compile(model, matrix: np.ndarray, dimension: int) -> dict:
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    raw = ridge.coef_.astype(np.float64) / scaler.scale_.astype(np.float64)
    intercept = float(ridge.intercept_ - raw @ scaler.mean_.astype(np.float64))
    current, source, difference, product = np.split(raw, 4)
    compiled = {
        "current": current + difference, "source": source - difference,
        "interaction": product, "intercept": intercept,
    }
    probe = matrix[: min(256, len(matrix))]
    c, s = probe[:, :dimension], probe[:, dimension:2 * dimension]
    score = intercept + c @ compiled["current"] + np.sum(
        s * (compiled["source"] + c * compiled["interaction"]), axis=1,
    )
    error = float(np.max(np.abs(score - model.predict(probe))))
    if error > 1e-8:
        raise RuntimeError(f"65-D exact MIPS compilation error {error}")
    compiled["maximum_verification_error"] = error
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-017_adaptation_context_address_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    cache_path = Path(config["output"]["candidate_cache"])
    artifact_path = Path(config["output"]["artifact"])
    if result_path.exists() or cache_path.exists() or artifact_path.exists():
        raise RuntimeError("EXP-017 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-017 requires train split and CUDA")
    prior_result = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior_result["experiment"] == "EXP-016" and prior_result["registered_gate"]["passed"] is False
        and prior_result["registered_gate"]["checks"]["random_component_ci_positive"] is False
        and all(prior_result["registered_gate"]["checks"][key] is True for key in (
            "positive_pooled_spearman", "positive_each_location_spearman", "minimum_policy_utility",
            "minimum_acceptance", "maximum_harm", "above_appearance",
        ))
        and prior_result["validation_accessed"] is False and prior_result["test_accessed"] is False
    ):
        raise RuntimeError("EXP-016 failure contract changed")
    prior_cache_path = Path(config["data"]["prior_candidate_cache"])
    if prior_result["candidate_cache_sha256"] != _sha256(prior_cache_path):
        raise RuntimeError("EXP-016 pair cache hash changed")
    prior = torch.load(prior_cache_path, map_location="cpu", weights_only=False)
    atom_result = json.loads(Path(config["model"]["atom_result"]).read_text())
    checkpoint_path = Path(config["model"]["atom_checkpoint"])
    if atom_result["checkpoint_sha256"] != _sha256(checkpoint_path) or atom_result["gate"]["passed"] is not True:
        raise RuntimeError("frozen EXP-015 atom contract failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["model"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    history = _adaptation_context(
        head, geometry, manifest, step_size=float(config["method"]["step_size"]), device=device,
    )
    tensor, target, metadata, target_table = _augment(prior, history)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-017", "protocol_revision": config["protocol_revision"], "split": "train",
        "features": tensor, "utility": target, "metadata": metadata, "target_table": target_table,
        "feature_contract": "65D=[visual64,self_improvement1]; pair=[c,s,c-s,c*s]",
        "query_or_future_online_input": False, "validation_accessed": False, "test_accessed": False,
    }, cache_path)
    matrix, utility = tensor.numpy().astype(np.float64), target.numpy().astype(np.float64)
    target_location = np.asarray([row["target_location"] for row in metadata])
    source_location = np.asarray([row["source_location"] for row in metadata])
    prediction, folds = _strict_oof(
        matrix, utility, target_location, source_location, float(config["address"]["ridge_alpha"]),
    )
    by_episode = {}
    for index, row in enumerate(metadata):
        by_episode.setdefault(row["episode"], []).append(index)
    policy, random_same_accept, appearance, oracle, accepted = {}, {}, {}, {}, {}
    threshold = float(config["method"]["acceptance_threshold"])
    for episode in target_table:
        indices = by_episode.get(episode, [])
        if not indices:
            policy[episode] = random_same_accept[episode] = appearance[episode] = oracle[episode] = 0.0
            accepted[episode] = False
            continue
        winner = max(indices, key=lambda index: (prediction[index], metadata[index]["source_context"]))
        take = bool(prediction[winner] > threshold)
        accepted[episode] = take
        policy[episode] = float(utility[winner]) if take else 0.0
        random_same_accept[episode] = float(np.mean(utility[indices])) if take else 0.0
        current_visual = matrix[indices[0], :64]
        appearance_winner = max(indices, key=lambda index: float(
            current_visual @ matrix[index, 65:129]
            / max(np.linalg.norm(current_visual) * np.linalg.norm(matrix[index, 65:129]), 1e-12)
        ))
        appearance[episode] = float(utility[appearance_winner]) if take else 0.0
        oracle[episode] = max(0.0, float(np.max(utility[indices])))
    policies = {
        "adaptation_context": _component_summary(policy, target_table, accepted),
        "random_same_accept": _component_summary(random_same_accept, target_table, accepted),
        "appearance_same_accept": _component_summary(appearance, target_table, accepted),
        "oracle_panel": _component_summary(oracle, target_table, {key: value > 0 for key, value in oracle.items()}),
    }
    bootstrap = {
        "primary_minus_random": _bootstrap(
            policy, random_same_accept, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "primary_minus_appearance": _bootstrap(
            policy, appearance, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    association = spearmanr(utility, prediction)
    success = config["success"]
    primary = policies["adaptation_context"]
    checks = {
        "positive_pooled_spearman": float(association.statistic) > 0,
        "positive_each_location_spearman": min(row["spearman"] for row in folds)
        > float(success["minimum_each_location_spearman"]),
        "minimum_policy_utility": primary["mean_utility"] > float(success["minimum_policy_utility"]),
        "minimum_acceptance": primary["acceptance"] >= float(success["minimum_acceptance"]),
        "maximum_harm": primary["harmful_rate"] <= float(success["maximum_harmful_rate"]),
        "above_appearance": primary["mean_utility"] > policies["appearance_same_accept"]["mean_utility"],
        "random_component_ci_positive": bootstrap["primary_minus_random"]["ci95"][0] > 0,
    }
    result = {
        "experiment": "EXP-017", "stage": "adaptation_context_address", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "targets": len(target_table), "pairs": len(metadata), "components": 25, "locations": 4,
        "source_entity_excluded": True, "query_or_future_online_input": False,
        "candidate_cache": str(cache_path), "candidate_cache_sha256": _sha256(cache_path),
        "pair_oof_spearman": float(association.statistic), "pair_oof_spearman_p": float(association.pvalue),
        "folds": folds, "policies": policies, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "validation_accessed": False, "test_accessed": False,
    }
    if all(checks.values()):
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["address"]["ridge_alpha"])))
        model.fit(matrix, utility)
        compiled = _compile(model, matrix, int(config["address"]["descriptor_dim"]))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "experiment": "EXP-017", "protocol_revision": config["protocol_revision"],
            "split": "train", "model": model, "compiled_mips": compiled,
            "feature_contract": "65D=[visual64,self_improvement1]; pair=[c,s,c-s,c*s]",
            "descriptor_dim": 65, "topk": 1, "acceptance_threshold": threshold,
            "atom_checkpoint": str(checkpoint_path), "atom_checkpoint_sha256": _sha256(checkpoint_path),
            "validation_accessed": False, "test_accessed": False,
        }, artifact_path)
        result["artifact"] = str(artifact_path)
        result["artifact_sha256"] = _sha256(artifact_path)
        result["mips_maximum_verification_error"] = compiled["maximum_verification_error"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "pair_oof_spearman": result["pair_oof_spearman"],
        "folds": folds, "policies": policies, "bootstrap": bootstrap,
        "gate": result["registered_gate"], "artifact": result.get("artifact"),
    }), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"EXP-017 adaptation-context gate failed: {checks}")


if __name__ == "__main__":
    main()
