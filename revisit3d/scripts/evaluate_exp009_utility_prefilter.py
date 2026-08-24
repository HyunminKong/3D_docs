#!/usr/bin/env python3
"""Component-crossfit observable utility prefilter for EXP-009 Stage 9."""

from __future__ import annotations

import argparse
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


def _columns(spec: list, dimensions: int) -> list[int]:
    spans = spec if spec and isinstance(spec[0], list) else [spec]
    result = []
    for start, stop in spans:
        result.extend(range(int(start), int(stop)))
    if not result or min(result) < 0 or max(result) >= dimensions:
        raise RuntimeError(f"invalid utility-prefilter feature spec {spec}")
    return result


def _fit_oof(
    matrix: np.ndarray,
    utility: np.ndarray,
    groups: np.ndarray,
    columns: list[int],
    alpha: float,
) -> np.ndarray:
    prediction = np.full(len(utility), np.nan, dtype=np.float64)
    for held_out in sorted(set(groups.tolist())):
        train, test = groups != held_out, groups == held_out
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(matrix[train][:, columns], utility[train])
        prediction[test] = model.predict(matrix[test][:, columns])
    if not np.isfinite(prediction).all():
        raise RuntimeError("utility prefilter did not produce complete OOF predictions")
    return prediction


def _random_expectation(
    rows_by_episode: dict[str, list[dict]],
    thresholds: dict[str, float],
    *,
    repetitions: int,
    candidate_count: int,
    seed: int,
    epsilon: float,
) -> tuple[dict[str, float], dict[str, float], dict]:
    episodes = sorted(rows_by_episode)
    oracle = np.zeros((repetitions, len(episodes)), dtype=np.float64)
    router = np.zeros_like(oracle)
    accepted = np.zeros_like(oracle, dtype=bool)
    for repetition in range(repetitions):
        generator = np.random.default_rng(seed + repetition)
        for episode_index, episode in enumerate(episodes):
            rows = rows_by_episode[episode]
            if not rows:
                continue
            indices = generator.choice(
                len(rows), size=min(candidate_count, len(rows)), replace=False,
            )
            chosen = [rows[int(index)] for index in indices]
            oracle[repetition, episode_index] = max(
                0.0, max(row["future_utility"] for row in chosen)
            )
            winner = max(chosen, key=lambda row: row["predicted_utility"])
            take = winner["predicted_utility"] > thresholds[winner["component"]]
            accepted[repetition, episode_index] = take
            router[repetition, episode_index] = winner["future_utility"] if take else 0.0
    return (
        {episode: float(oracle[:, index].mean()) for index, episode in enumerate(episodes)},
        {episode: float(router[:, index].mean()) for index, episode in enumerate(episodes)},
        {
            "oracle_mean_utility": float(oracle.mean()),
            "router_mean_utility": float(router.mean()),
            "router_harm_median": float(np.median((router < -epsilon).mean(axis=1))),
            "router_acceptance_mean": float(accepted.mean()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_utility_prefilter_v19.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-009 Stage-9 result already exists: {output}")
    candidate = json.loads(Path(config["source"]["candidate_cache"]).read_text())
    stage8 = json.loads(Path(config["source"]["stage8_result"]).read_text())
    stage7 = json.loads(Path(config["source"]["stage7_result"]).read_text())
    stage7_config = yaml.safe_load(Path(config["source"]["stage7_config"]).read_text())
    stage8_config = yaml.safe_load(Path(config["source"]["stage8_config"]).read_text())
    if not (
        candidate.get("split") == stage8.get("split") == stage7.get("split") == "train"
        and candidate.get("validation_accessed") is False
        and stage8.get("validation_accessed") is False
        and stage7.get("validation_accessed") is False
        and candidate.get("test_accessed") is False
        and stage8.get("test_accessed") is False
        and stage7.get("test_accessed") is False
        and candidate.get("query_or_future_router_input") is False
        and stage8.get("query_or_future_router_input") is False
        and stage7.get("query_or_future_router_input") is False
        and candidate.get("prefilter_feature_dimensions") == 274
    ):
        raise RuntimeError("Stage 9 requires locked train-only observable pair features")

    rows = candidate["rows"]
    matrix = np.asarray([row["prefilter_features"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["component"] for row in rows])
    if matrix.shape != (len(rows), 274) or len(set(groups.tolist())) != 25:
        raise RuntimeError("Stage-9 candidate matrix/component contract changed")
    rows_by_episode: dict[str, list[dict]] = {}
    indices_by_episode: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        rows_by_episode.setdefault(row["episode"], []).append(row)
        indices_by_episode.setdefault(row["episode"], []).append(index)
    if len(rows_by_episode) != 218:
        raise RuntimeError("Stage 9 requires all 218 unique causal targets")
    group_by_episode = {
        episode: subset[0]["component"] for episode, subset in rows_by_episode.items()
    }
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

    variants = {}
    variant_values = {}
    alpha = float(config["prefilter"]["ridge_alpha"])
    candidate_count = int(config["prefilter"]["candidate_count"])
    for name, spec in config["prefilter"]["feature_variants"].items():
        columns = _columns(spec, matrix.shape[1])
        prediction = _fit_oof(matrix, utility, groups, columns, alpha)
        oracle_values, oracle_accept = {}, {}
        router_values, router_accept = {}, {}
        top1_values, top1_accept = {}, {}
        for episode, indices in indices_by_episode.items():
            ranked = sorted(indices, key=lambda index: (-prediction[index], rows[index]["candidate_context"]))
            topk = ranked[:candidate_count]
            oracle_value = max(0.0, max(utility[index] for index in topk))
            top1_value = float(utility[topk[0]])
            router_choice = max(topk, key=lambda index: rows[index]["predicted_utility"])
            take = rows[router_choice]["predicted_utility"] > thresholds[rows[router_choice]["component"]]
            oracle_values[episode] = float(oracle_value)
            oracle_accept[episode] = oracle_value > 0.0
            top1_values[episode] = top1_value
            top1_accept[episode] = True
            router_values[episode] = float(utility[router_choice]) if take else 0.0
            router_accept[episode] = bool(take)
        association = spearmanr(utility, prediction)
        variants[name] = {
            "feature_dimensions": len(columns),
            "candidate_oof_spearman": float(association.statistic),
            "candidate_oof_spearman_p": float(association.pvalue),
            "top1": _summarize(top1_values, top1_accept, group_by_episode, epsilon),
            "oracle_topk": _summarize(oracle_values, oracle_accept, group_by_episode, epsilon),
            "router": _summarize(router_values, router_accept, group_by_episode, epsilon),
        }
        variant_values[name] = {
            "oracle": oracle_values, "router": router_values,
        }
        print(json.dumps({"variant": name, **variants[name]}), flush=True)

    primary = config["prefilter"]["primary_features"]
    if primary != "all_observable" or primary not in variants:
        raise RuntimeError("Stage-9 registered primary feature set changed")
    bootstrap = {
        "primary_minus_random_expected_oracle_topk": _paired_component_bootstrap(
            variant_values[primary]["oracle"], random_oracle, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "primary_minus_random_expected_router": _paired_component_bootstrap(
            variant_values[primary]["router"], random_router, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    primary_metrics = variants[primary]
    checks = {
        "positive_oof_spearman": primary_metrics["candidate_oof_spearman"] > 0.0,
        "oracle_component_interval_positive":
            bootstrap["primary_minus_random_expected_oracle_topk"]["ci95"][0] > 0.0,
        "router_component_interval_positive":
            bootstrap["primary_minus_random_expected_router"]["ci95"][0] > 0.0,
        "router_harm_not_above_random_median":
            primary_metrics["router"]["harmful_rate"] <= random_summary["router_harm_median"],
    }
    result = {
        "experiment": "EXP-009", "stage": "stage9_utility_supervised_prefilter",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "targets": len(rows_by_episode), "candidates": len(rows),
        "components": len(set(groups.tolist())), "primary": primary,
        "random_expected": random_summary,
        "stage7_dinov2": stage7["metrics"]["dinov2"],
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
