#!/usr/bin/env python3
"""Parameter-free agreement reranking with coarse top-1 fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.scripts.fit_exp016_unified_utility_address import (
    _bootstrap,
    _compile,
    _component_summary,
    _sha256,
    _strict_oof,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-019_agreement_fallback_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    artifact_path = Path(config["output"]["artifact"])
    if result_path.exists() or artifact_path.exists():
        raise RuntimeError("EXP-019 output already exists")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior["experiment"] == "EXP-018" and prior["registered_gate"]["passed"] is False
        and prior["registered_gate"]["checks"]["minimum_policy_utility"] is False
        and prior["registered_gate"]["checks"]["coarse_component_ci_positive"] is False
        and prior["registered_gate"]["checks"]["random_component_ci_positive"] is True
        and prior["validation_accessed"] is False and prior["test_accessed"] is False
    ):
        raise RuntimeError("EXP-018 failure contract changed")
    pair_path = Path(config["data"]["candidate_cache"])
    agreement_path = Path(config["data"]["agreement_cache"])
    if not (
        prior["agreement_cache_sha256"] == _sha256(agreement_path)
        and torch.load(agreement_path, map_location="cpu", weights_only=False)["candidate_cache_sha256"]
        == _sha256(pair_path)
    ):
        raise RuntimeError("EXP-018 cache hash contract failed")
    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    agreement_payload = torch.load(agreement_path, map_location="cpu", weights_only=False)
    matrix = pair["features"].numpy().astype(np.float64)
    utility = pair["utility"].numpy().astype(np.float64)
    agreement = agreement_payload["current_geometry_agreement"].numpy().astype(np.float64)
    metadata, target_table = pair["metadata"], pair["target_table"]
    if len(agreement) != len(utility) or agreement_payload["metadata"] != metadata:
        raise RuntimeError("agreement/pair row order changed")
    target_location = np.asarray([row["target_location"] for row in metadata])
    source_location = np.asarray([row["source_location"] for row in metadata])
    prediction, folds = _strict_oof(
        matrix, utility, target_location, source_location, float(config["address"]["ridge_alpha"]),
    )
    by_episode = {}
    for index, row in enumerate(metadata):
        by_episode.setdefault(row["episode"], []).append(index)
    full, coarse, random_same_accept, accepted = {}, {}, {}, {}
    safe_reroutes, fallbacks = 0, 0
    k = int(config["method"]["coarse_topk"])
    utility_threshold = float(config["method"]["utility_threshold"])
    agreement_threshold = float(config["method"]["geometry_agreement_threshold"])
    for episode in target_table:
        indices = by_episode.get(episode, [])
        if not indices:
            full[episode] = coarse[episode] = random_same_accept[episode] = 0.0
            accepted[episode] = False
            continue
        ranked = sorted(indices, key=lambda index: (-prediction[index], metadata[index]["source_context"]))
        coarse_winner = ranked[0]
        topk = ranked[: min(k, len(ranked))]
        safe = [
            index for index in topk
            if prediction[index] > utility_threshold and agreement[index] > agreement_threshold
        ]
        if safe:
            winner = max(safe, key=lambda index: (agreement[index], metadata[index]["source_context"]))
            safe_reroutes += int(winner != coarse_winner)
        else:
            winner = coarse_winner
            fallbacks += 1
        take = bool(prediction[winner] > utility_threshold)
        accepted[episode] = take
        full[episode] = float(utility[winner]) if take else 0.0
        coarse_take = bool(prediction[coarse_winner] > utility_threshold)
        coarse[episode] = float(utility[coarse_winner]) if coarse_take else 0.0
        random_same_accept[episode] = float(np.mean(utility[indices])) if take else 0.0
    policies = {
        "agreement_fallback": _component_summary(full, target_table, accepted),
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
    primary = policies["agreement_fallback"]
    success = config["success"]
    checks = {
        "minimum_policy_utility": primary["mean_utility"] > float(success["minimum_policy_utility"]),
        "minimum_acceptance": primary["acceptance"] >= float(success["minimum_acceptance"]),
        "maximum_harm": primary["harmful_rate"] <= float(success["maximum_harmful_rate"]),
        "random_component_ci_positive": bootstrap["full_minus_random"]["ci95"][0] > 0,
        "coarse_component_ci_positive": bootstrap["full_minus_coarse"]["ci95"][0] > 0,
    }
    result = {
        "experiment": "EXP-019", "stage": "agreement_fallback", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "targets": len(target_table), "pairs": len(metadata), "components": 25,
        "coarse_topk": k, "safe_reroutes": safe_reroutes, "coarse_fallbacks": fallbacks,
        "learned_fine_router": False, "future_or_query_online_input": False,
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
            "experiment": "EXP-019", "protocol_revision": config["protocol_revision"],
            "split": "train", "coarse_model": model, "compiled_mips": compiled,
            "descriptor_dim": 64, "topk": k, "utility_threshold": utility_threshold,
            "fine_rule": "positive_geometry_argmax_else_coarse_top1",
            "geometry_agreement_threshold": agreement_threshold,
            "validation_accessed": False, "test_accessed": False,
        }, artifact_path)
        result["artifact"] = str(artifact_path)
        result["artifact_sha256"] = _sha256(artifact_path)
        result["mips_maximum_verification_error"] = compiled["maximum_verification_error"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "policies": policies, "bootstrap": bootstrap,
        "safe_reroutes": safe_reroutes, "fallbacks": fallbacks,
        "gate": result["registered_gate"], "artifact": result.get("artifact"),
    }), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"EXP-019 agreement-fallback gate failed: {checks}")


if __name__ == "__main__":
    main()
