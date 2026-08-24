#!/usr/bin/env python3
"""Simulate fold-local causal banks using leakage-safe EXP-007 OOF tables."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _add, _evaluate, _orders, _sha256


def _summary(rows: list[dict], epsilon: float, group_by_episode: dict[str, str]) -> dict:
    methods = (
        "router_topk", "router_all", "appearance_top1", "random_expectation",
        "oracle_topk", "oracle_all",
    )
    metrics = {}
    for method in methods:
        values = np.asarray([row[method] for row in rows])
        component_order = []
        for fold_order in sorted({(row["fold"], row["order"]) for row in rows}):
            fold, order = fold_order
            for group in sorted(set(group_by_episode.values())):
                subset = [
                    row[method] for row in rows
                    if row["fold"] == fold and row["order"] == order
                    and group_by_episode[row["episode"]] == group
                ]
                if subset:
                    component_order.append(float(np.mean(subset)))
        metrics[method] = {
            "mean_utility": float(values.mean()),
            "median_utility": float(np.median(values)),
            "beneficial_rate": float(np.mean(values > epsilon)),
            "harmful_rate": float(np.mean(values < -epsilon)),
            "raw_sign_harm_rate": float(np.mean(values < 0)),
            "component_order_harm_rate": float(np.mean(np.asarray(component_order) < -epsilon)),
            "mean_regret_to_causal_unbounded_oracle": float(np.mean([
                row["causal_unbounded_oracle"] - row[method] for row in rows
            ])),
        }
    return {
        "events": len(rows),
        "metrics": metrics,
        "router_accept_rate": float(np.mean([row["router_accepted"] for row in rows])),
        "oracle_topk_recall": float(np.mean([row["oracle_in_topk"] for row in rows])),
        "mean_bank_records": float(np.mean([row["bank_records"] for row in rows])),
        "max_bank_records": max(row["bank_records"] for row in rows),
        "mean_unique_contexts": float(np.mean([row["bank_unique_contexts"] for row in rows])),
        "appearance_comparisons": sum(row["appearance_comparisons"] for row in rows),
        "router_comparisons": sum(row["router_comparisons"] for row in rows),
    }


def _run_variant(
    streams: list[dict], order_names: list[str], policy: str, capacity: int | None,
    top_k: int, threshold: float, seed: int, variant_index: int,
) -> tuple[list[dict], dict, list[int]]:
    rows = []
    aggregate = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
    final_sizes = []
    for stream in streams:
        contexts = {row["context_id"]: row for row in stream["contexts"]}
        lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
        orders = _orders(order_names, stream["events"])
        for order_index, (order_name, sequence) in enumerate(orders.items()):
            bank = []
            counts = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
            seen = [0]
            rng = random.Random(seed + 1009 * variant_index + 97 * int(stream["fold"]) + order_index)
            for position, event_index in enumerate(sequence):
                event = stream["events"][event_index]
                for context_id in event["pre_query_writes"]:
                    _add(
                        bank, contexts[context_id], policy=policy, capacity=capacity,
                        rng=rng, counters=counts, seen_writes=seen,
                    )
                evaluation = _evaluate(bank, event["episode"], lookup, top_k, threshold)
                rows.append({
                    "fold": stream["fold"], "policy": policy, "capacity": capacity,
                    "order": order_name, "stream_position": position,
                    "episode": event["episode"], **evaluation,
                })
                for context_id in event["post_query_writes"]:
                    _add(
                        bank, contexts[context_id], policy=policy, capacity=capacity,
                        rng=rng, counters=counts, seen_writes=seen,
                    )
            final_sizes.append(len(bank))
            for key in aggregate:
                aggregate[key] += counts[key]
    return rows, aggregate, final_sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_oof_v12.yaml")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["stage0"]["simulation_result"])
    if output.exists():
        raise RuntimeError(f"EXP-007 v1.2 simulation already exists: {output}")
    table_path = Path(config["stage0"]["utility_table"])
    table = json.loads(table_path.read_text())
    if not (
        table.get("protocol_revision") == "v1.2"
        and table.get("split") == "train"
        and table.get("validation_accessed") is False
        and table.get("fold_local_memory_coordinates") is True
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("simulation requires the leakage-safe fold-local v1.2 table")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {record["episode_id"]: groups[index] for index, record in enumerate(records)}
    variants = [("unbounded_all_write", None), ("unbounded_unique_context", None)] + [
        (policy, int(capacity))
        for policy in config["stage0"]["policies"]
        for capacity in config["stage0"]["capacities"]
    ]
    run_rows = {}
    counts_by_variant = {}
    final_sizes_by_variant = {}
    for variant_index, (policy, capacity) in enumerate(variants):
        key = f"{policy}:{capacity}"
        rows, counts, final_sizes = _run_variant(
            table["streams"], config["stage0"]["stream_orders"], policy, capacity,
            int(config["stage0"]["top_k"]), float(config["router"]["utility_threshold"]),
            int(config["seed"]), variant_index,
        )
        run_rows[key] = rows
        counts_by_variant[key] = counts
        final_sizes_by_variant[key] = final_sizes

    reference_rows = run_rows["unbounded_all_write:None"]
    oracle_by_event = {
        (row["fold"], row["order"], row["episode"]): row["oracle_all"]
        for row in reference_rows
    }
    variant_results = []
    all_rows = []
    for policy, capacity in variants:
        key = f"{policy}:{capacity}"
        rows = run_rows[key]
        for row in rows:
            row["causal_unbounded_oracle"] = oracle_by_event[
                (row["fold"], row["order"], row["episode"])
            ]
        summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variant_results.append({
            "policy": policy,
            "capacity": capacity,
            "summary": summary,
            "counts": counts_by_variant[key],
            "mean_final_records": float(np.mean(final_sizes_by_variant[key])),
        })
        all_rows.extend(rows)
        print(json.dumps({
            "policy": policy, "capacity": capacity,
            "router_topk": summary["metrics"]["router_topk"],
            "max_bank": summary["max_bank_records"],
        }), flush=True)

    unbounded = variant_results[0]
    reference = unbounded["summary"]["metrics"]["router_topk"]
    promising = []
    for variant in variant_results:
        if variant["capacity"] is None:
            continue
        metric = variant["summary"]["metrics"]["router_topk"]
        retention = (
            metric["mean_utility"] / reference["mean_utility"]
            if reference["mean_utility"] > 0 else float("nan")
        )
        variant["retention_vs_unbounded_router"] = retention
        variant["harm_delta_vs_unbounded"] = metric["harmful_rate"] - reference["harmful_rate"]
        passed = (
            variant["capacity"] <= int(config["stage0"]["maximum_success_capacity"])
            and retention >= float(config["stage0"]["retention_target"])
            and metric["harmful_rate"] <= reference["harmful_rate"]
        )
        variant["registered_promising_policy"] = bool(passed)
        if passed:
            promising.append({"policy": variant["policy"], "capacity": variant["capacity"]})

    result = {
        "experiment": "EXP-007",
        "stage": "stage0_oof_fold_local_causal_capacity",
        "split": "train",
        "protocol_revision": "v1.2",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "offline_utility_label_only",
        "fold_local_memory_coordinates": True,
        "cross_fold_bank_mixing": False,
        "pseudo_stream_not_real_temporal_claim": True,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "utility_table": str(table_path),
        "utility_table_sha256": _sha256(table_path),
        "variants": variant_results,
        "registered_gate": {
            "retention_target": float(config["stage0"]["retention_target"]),
            "maximum_capacity": int(config["stage0"]["maximum_success_capacity"]),
            "no_harm_increase": True,
            "promising_variants": promising,
            "passed": bool(promising),
        },
        "runtime_s": time.perf_counter() - started,
        "rows": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "gate": result["registered_gate"],
        "unbounded": unbounded["summary"], "runtime_s": result["runtime_s"],
    }), flush=True)


if __name__ == "__main__":
    main()
