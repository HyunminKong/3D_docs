#!/usr/bin/env python3
"""Location-held-out, source-entity-safe utility prefilter correction for EXP-009."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.scripts.evaluate_exp006_validation import _summarize
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _paired_component_bootstrap
from revisit3d.scripts.evaluate_exp009_utility_prefilter import _columns, _random_expectation


def _identifier(segment: dict) -> str:
    payload = f"{segment['scene']}:{','.join(map(str, segment['frames']))}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _strict_location_oof(
    matrix: np.ndarray,
    utility: np.ndarray,
    target_location: np.ndarray,
    source_location: np.ndarray,
    columns: list[int],
    alpha: float,
) -> tuple[np.ndarray, list[dict]]:
    prediction = np.full(len(utility), np.nan, dtype=np.float64)
    folds = []
    for held_out in sorted(set(target_location.tolist())):
        train = (target_location != held_out) & (source_location != held_out)
        test = target_location == held_out
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(matrix[train][:, columns], utility[train])
        prediction[test] = model.predict(matrix[test][:, columns])
        association = spearmanr(utility[test], prediction[test])
        folds.append({
            "held_out_location": held_out,
            "strict_train_pairs": int(train.sum()), "test_pairs": int(test.sum()),
            "spearman": float(association.statistic), "spearman_p": float(association.pvalue),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("strict location crossfit did not cover every candidate")
    return prediction, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_location_crossfit_prefilter_v20.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-009 Stage-10 result already exists: {output}")
    candidate = json.loads(Path(config["source"]["candidate_cache"]).read_text())
    stage7 = json.loads(Path(config["source"]["stage7_result"]).read_text())
    stage8 = json.loads(Path(config["source"]["stage8_result"]).read_text())
    stage9 = json.loads(Path(config["source"]["stage9_result"]).read_text())
    stage7_config = yaml.safe_load(Path(config["source"]["stage7_config"]).read_text())
    stage8_config = yaml.safe_load(Path(config["source"]["stage8_config"]).read_text())
    if not (
        candidate.get("split") == stage7.get("split") == stage8.get("split")
        == stage9.get("split") == "train"
        and all(payload.get("validation_accessed") is False for payload in (
            candidate, stage7, stage8, stage9,
        ))
        and all(payload.get("test_accessed") is False for payload in (
            candidate, stage7, stage8, stage9,
        ))
        and all(payload.get("query_or_future_router_input") is False for payload in (
            candidate, stage7, stage8, stage9,
        ))
        and candidate.get("prefilter_feature_dimensions") == 274
    ):
        raise RuntimeError("Stage 10 requires locked train-only observable artifacts")

    context_location = {}
    manifest = json.loads(Path(config["source"]["geometry_manifest"]).read_text())
    for row in manifest:
        for tag in ("a", "b", "a_prime"):
            key = _identifier(row[tag])
            previous = context_location.setdefault(key, row["location"])
            if previous != row["location"]:
                raise RuntimeError("context crosses official locations")
    if len(context_location) != 557:
        raise RuntimeError("location audit must cover all 557 causal contexts")

    rows = candidate["rows"]
    matrix = np.asarray([row["prefilter_features"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    target_location = np.asarray([context_location[row["target_context"]] for row in rows])
    source_location = np.asarray([context_location[row["candidate_context"]] for row in rows])
    if len(set(target_location.tolist())) != 4 or matrix.shape != (len(rows), 274):
        raise RuntimeError("location/prefilter matrix contract changed")
    rows_by_episode, indices_by_episode = {}, {}
    for index, row in enumerate(rows):
        rows_by_episode.setdefault(row["episode"], []).append(row)
        indices_by_episode.setdefault(row["episode"], []).append(index)
    group_by_episode = {
        row["episode"]: row["component"] for row in stage7["selection_rows"]
    }
    for episode in group_by_episode:
        rows_by_episode.setdefault(episode, [])
        indices_by_episode.setdefault(episode, [])
    if len(rows_by_episode) != 218:
        raise RuntimeError("Stage 10 requires all 218 causal targets")

    stage6 = json.loads(Path(stage7_config["data"]["stage6_result"]).read_text())
    thresholds = {
        component: float(value["threshold"])
        for component, value in stage6["threshold_by_component"].items()
    }
    epsilon = float(config["statistics"]["utility_deadband"])
    random_oracle, random_router, random_summary = _random_expectation(
        rows_by_episode, thresholds,
        repetitions=int(stage8_config["random_null"]["repetitions"]),
        candidate_count=int(config["prefilter"]["candidate_count"]),
        seed=int(stage8_config["seed"]), epsilon=epsilon,
    )

    variants, values = {}, {}
    alpha = float(config["prefilter"]["ridge_alpha"])
    candidate_count = int(config["prefilter"]["candidate_count"])
    for name, spec in config["prefilter"]["feature_variants"].items():
        columns = _columns(spec, matrix.shape[1])
        prediction, folds = _strict_location_oof(
            matrix, utility, target_location, source_location, columns, alpha,
        )
        association = spearmanr(utility, prediction)
        oracle_values, oracle_accept = {}, {}
        router_values, router_accept = {}, {}
        top1_values, top1_accept = {}, {}
        for episode, indices in indices_by_episode.items():
            if not indices:
                oracle_values[episode] = router_values[episode] = top1_values[episode] = 0.0
                oracle_accept[episode] = router_accept[episode] = top1_accept[episode] = False
                continue
            ranked = sorted(
                indices, key=lambda index: (-prediction[index], rows[index]["candidate_context"]),
            )
            topk = ranked[:candidate_count]
            oracle_value = max(0.0, max(utility[index] for index in topk))
            router_choice = max(topk, key=lambda index: rows[index]["predicted_utility"])
            take = rows[router_choice]["predicted_utility"] > thresholds[rows[router_choice]["component"]]
            oracle_values[episode] = float(oracle_value)
            oracle_accept[episode] = oracle_value > 0.0
            top1_values[episode] = float(utility[topk[0]])
            top1_accept[episode] = True
            router_values[episode] = float(utility[router_choice]) if take else 0.0
            router_accept[episode] = bool(take)
        variants[name] = {
            "feature_dimensions": len(columns),
            "candidate_oof_spearman": float(association.statistic),
            "candidate_oof_spearman_p": float(association.pvalue),
            "folds": folds,
            "top1": _summarize(top1_values, top1_accept, group_by_episode, epsilon),
            "oracle_topk": _summarize(oracle_values, oracle_accept, group_by_episode, epsilon),
            "router": _summarize(router_values, router_accept, group_by_episode, epsilon),
        }
        values[name] = {"oracle": oracle_values, "router": router_values}
        print(json.dumps({"variant": name, **variants[name]}), flush=True)

    primary = config["prefilter"]["primary_features"]
    if primary != "transport_descriptor" or primary not in variants:
        raise RuntimeError("Stage-10 registered primary changed")
    bootstrap = {
        "primary_minus_random_expected_oracle_topk": _paired_component_bootstrap(
            values[primary]["oracle"], random_oracle, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "primary_minus_random_expected_router": _paired_component_bootstrap(
            values[primary]["router"], random_router, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    primary_metrics = variants[primary]
    checks = {
        "positive_pooled_oof_spearman": primary_metrics["candidate_oof_spearman"] > 0.0,
        "positive_each_location_spearman": min(
            row["spearman"] for row in primary_metrics["folds"]
        ) > float(config["success"]["minimum_each_location_spearman"]),
        "oracle_component_interval_positive":
            bootstrap["primary_minus_random_expected_oracle_topk"]["ci95"][0] > 0.0,
        "router_component_interval_positive":
            bootstrap["primary_minus_random_expected_router"]["ci95"][0] > 0.0,
        "router_harm_not_above_random_median":
            primary_metrics["router"]["harmful_rate"] <= random_summary["router_harm_median"],
    }
    result = {
        "experiment": "EXP-009", "stage": "stage10_location_crossfit_utility_prefilter",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "targets": len(rows_by_episode), "candidates": len(rows),
        "locations": sorted(set(target_location.tolist())), "primary": primary,
        "source_entity_excluded": True, "random_expected": random_summary,
        "variants": variants, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "primary": primary_metrics,
        "random_expected": random_summary, "bootstrap": bootstrap,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
