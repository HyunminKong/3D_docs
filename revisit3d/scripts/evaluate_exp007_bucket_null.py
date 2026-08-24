#!/usr/bin/env python3
"""Matched permutation-null test for the EXP-007 crossfit consolidation key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _evaluate, _orders, _sha256
from revisit3d.scripts.simulate_exp007_token_bucket import _base_policy, _write_token
from revisit3d.scripts.simulate_exp007_utility_consolidation import _summary, _update_history


def _probability_map(pair_rows: list[dict], values: np.ndarray) -> dict[int, dict[tuple[str, str], float]]:
    output: dict[int, dict[tuple[str, str], float]] = {}
    for row, value in zip(pair_rows, values):
        fold = int(row["fold"])
        output.setdefault(fold, {})[tuple(sorted((row["left"], row["right"])))] = float(value)
    return output


def _simulate(
    table: dict, oracle_reference: dict, probability: dict, config: dict,
    group_by_episode: dict[str, str], policy: str,
) -> dict:
    capacity = int(config["bank"]["capacity"])
    top_k = int(config["bank"]["top_k"])
    threshold = float(config["bank"]["probability_threshold"])
    rows = []
    counts = {
        "writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0,
        "true_bucket_merges": 0, "false_bucket_merges": 0,
    }
    for stream in table["streams"]:
        fold = int(stream["fold"])
        contexts = {row["context_id"]: row for row in stream["contexts"]}
        lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
        orders = _orders(config["bank"]["stream_orders"], stream["events"])
        for order_index, (order_name, sequence) in enumerate(orders.items()):
            bank = []
            serial = int(config["seed"]) + 100000 * fold + 1000 * order_index
            for position, event_index in enumerate(sequence):
                event = stream["events"][event_index]
                for context_id in event["pre_query_writes"]:
                    serial += 1
                    _write_token(
                        bank, contexts[context_id], policy, capacity, serial,
                        probability[fold], threshold, config["bank"], counts,
                    )
                evaluation = _evaluate(bank, event["episode"], lookup, top_k, 0.0)
                evaluation["causal_unbounded_oracle"] = oracle_reference[
                    (fold, order_name, event["episode"])
                ]
                rows.append({
                    "fold": fold, "policy": policy, "capacity": capacity,
                    "order": order_name, "stream_position": position,
                    "episode": event["episode"], **evaluation,
                })
                _update_history(
                    bank, event["episode"], lookup, _base_policy(policy), top_k,
                    evaluation["selected_context"],
                )
                for context_id in event["post_query_writes"]:
                    serial += 1
                    _write_token(
                        bank, contexts[context_id], policy, capacity, serial,
                        probability[fold], threshold, config["bank"], counts,
                    )
    return _summary(rows, float(table["utility_deadband"]), group_by_episode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_crossfit_bucket_null_v21.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"bucket-null result already exists: {output}")

    table_path = Path(config["source"]["oof_utility_table"])
    capacity_path = Path(config["source"]["oof_capacity_result"])
    bucket_path = Path(config["source"]["frozen_bucket_result"])
    table = json.loads(table_path.read_text())
    capacity = json.loads(capacity_path.read_text())
    bucket = json.loads(bucket_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and capacity.get("validation_accessed") is False
        and bucket.get("validation_accessed") is False
        and bucket.get("protocol_revision") == "v2.0"
        and bucket.get("bucket_key_source") == "frozen_foundation_crossfit_pca"
    ):
        raise RuntimeError("v2.1 requires the locked train-only v2.0 crossfit result")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {row["episode_id"]: groups[index] for index, row in enumerate(records)}
    oracle_rows = [
        row for row in capacity["rows"]
        if row["policy"] == "unbounded_all_write" and row["capacity"] is None
    ]
    oracle_reference = {
        (int(row["fold"]), row["order"], row["episode"]): row["causal_unbounded_oracle"]
        for row in oracle_rows
    }

    pair_rows = bucket["pair_rows"]
    original = np.asarray([row["oof_probability"] for row in pair_rows], dtype=np.float64)
    pair_fold = np.asarray([int(row["fold"]) for row in pair_rows])
    rng = np.random.default_rng(int(config["seed"]))
    draws = {policy: [] for policy in config["bank"]["policies"]}
    for permutation in range(int(config["permutation_null"]["permutations"])):
        shuffled = original.copy()
        for fold in sorted(set(pair_fold.tolist())):
            mask = np.flatnonzero(pair_fold == fold)
            shuffled[mask] = rng.permutation(shuffled[mask])
        probability = _probability_map(pair_rows, shuffled)
        for policy in config["bank"]["policies"]:
            metric = _simulate(
                table, oracle_reference, probability, config, group_by_episode, policy,
            )
            draws[policy].append({
                "mean_utility": metric["mean_utility"],
                "harmful_rate": metric["harmful_rate"],
                "mean_regret": metric["mean_regret_to_causal_unbounded_oracle"],
            })
        if (permutation + 1) % 100 == 0:
            print(json.dumps({"completed": permutation + 1}), flush=True)

    observed = {row["policy"]: row for row in bucket["variants"]}
    policy_results = []
    for policy, samples in draws.items():
        utility = np.asarray([row["mean_utility"] for row in samples])
        harm = np.asarray([row["harmful_rate"] for row in samples])
        observed_metric = observed[policy]["summary"]
        p_value = float((1 + (utility >= observed_metric["mean_utility"]).sum()) / (len(utility) + 1))
        policy_results.append({
            "policy": policy,
            "observed": observed_metric,
            "null": {
                "samples": len(samples),
                "mean_utility_mean": float(utility.mean()),
                "mean_utility_q025_q50_q975": [float(value) for value in np.quantile(utility, [.025, .5, .975])],
                "harmful_rate_mean": float(harm.mean()),
                "harmful_rate_q025_q50_q975": [float(value) for value in np.quantile(harm, [.025, .5, .975])],
            },
            "one_sided_utility_p": p_value,
            "observed_utility_percentile": float((utility < observed_metric["mean_utility"]).mean()),
        })

    primary = next(row for row in policy_results if row["policy"] == config["bank"]["primary_policy"])
    appearance = bucket["appearance_diversity_capacity8"]
    passed = (
        primary["one_sided_utility_p"] < float(config["success"]["one_sided_alpha"])
        and primary["observed"]["harmful_rate"] <= appearance["harmful_rate"]
    )
    result = {
        "experiment": "EXP-007",
        "stage": "stage9_crossfit_bucket_permutation_null",
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "validation_accessed": False,
        "test_accessed": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_bucket_result": str(bucket_path),
        "null_contract": "shuffle OOF pair probabilities within fold; preserve score distribution",
        "appearance_diversity_capacity8": appearance,
        "policies": policy_results,
        "registered_gate": {
            "primary_policy": config["bank"]["primary_policy"],
            "passed": bool(passed),
            "one_sided_alpha": float(config["success"]["one_sided_alpha"]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
