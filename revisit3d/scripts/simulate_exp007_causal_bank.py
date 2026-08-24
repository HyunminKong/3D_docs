#!/usr/bin/env python3
"""Simulate causal capacity-bounded EXP-007 banks from the frozen utility table."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _orders(names: list[str], events: list[dict]) -> dict[str, list[int]]:
    indices = list(range(len(events)))
    by_episode = sorted(indices, key=lambda index: events[index]["episode"])
    result = {}
    for name in names:
        if name == "lexicographic":
            order = by_episode
        elif name == "reverse_lexicographic":
            order = list(reversed(by_episode))
        elif name.startswith("random_"):
            order = list(indices)
            random.Random(int(name.removeprefix("random_"))).shuffle(order)
        else:
            raise ValueError(f"unsupported stream order {name!r}")
        result[name] = order
    return result


def _add(
    bank: list[dict], context: dict, *, policy: str, capacity: int | None,
    rng: random.Random, counters: dict[str, int], seen_writes: list[int],
) -> None:
    counters["writes"] += 1
    seen_writes[0] += 1
    entry = {
        "context_id": context["context_id"],
        "scene": context["scene"],
        "serial": seen_writes[0],
        "descriptor": context["descriptor"],
    }
    if policy == "unbounded_all_write":
        bank.append(entry)
        return
    if policy == "unbounded_unique_context":
        if any(item["context_id"] == entry["context_id"] for item in bank):
            counters["merges"] += 1
        else:
            bank.append(entry)
        return
    if capacity is None:
        raise ValueError(f"bounded policy {policy} requires capacity")
    if policy == "fifo":
        bank.append(entry)
        if len(bank) > capacity:
            bank.pop(0)
            counters["evictions"] += 1
        return
    if policy == "reservoir":
        if len(bank) < capacity:
            bank.append(entry)
        else:
            position = rng.randrange(seen_writes[0])
            if position < capacity:
                bank[position] = entry
                counters["evictions"] += 1
            else:
                counters["rejected_writes"] += 1
        return
    if policy == "scene_latest":
        matches = [index for index, item in enumerate(bank) if item["scene"] == entry["scene"]]
        if matches:
            for index in reversed(matches):
                bank.pop(index)
            counters["merges"] += len(matches)
        bank.append(entry)
        if len(bank) > capacity:
            bank.pop(0)
            counters["evictions"] += 1
        return
    if policy == "appearance_diversity":
        duplicate = next(
            (index for index, item in enumerate(bank) if item["context_id"] == entry["context_id"]),
            None,
        )
        if duplicate is not None:
            # An exactly repeated context adds frequency evidence but no new atom.
            bank[duplicate]["serial"] = entry["serial"]
            counters["merges"] += 1
            return
        if len(bank) < capacity:
            bank.append(entry)
            return
        candidates = bank + [entry]
        descriptor = np.asarray([item["descriptor"] for item in candidates], dtype=np.float64)
        descriptor /= np.linalg.norm(descriptor, axis=1, keepdims=True).clip(1e-12)
        similarity = descriptor @ descriptor.T
        np.fill_diagonal(similarity, -np.inf)
        redundancy = similarity.max(axis=1)
        most_redundant = np.flatnonzero(np.isclose(redundancy, redundancy.max()))
        # On ties remove the oldest item; the new observation is retained only
        # if it contributes genuine descriptor diversity.
        remove = min(most_redundant, key=lambda index: candidates[index]["serial"])
        if remove == len(bank):
            counters["rejected_writes"] += 1
        else:
            bank[remove] = entry
            counters["evictions"] += 1
        return
    raise ValueError(f"unsupported bank policy {policy!r}")


def _evaluate(
    bank: list[dict], episode: str, lookup: dict[tuple[str, str], dict], top_k: int,
    threshold: float,
) -> dict:
    if not bank:
        raise RuntimeError("bank must contain the within-event A/B observations")
    rows = [lookup[(episode, item["context_id"])] for item in bank]
    appearance = np.asarray([row["appearance_similarity"] for row in rows])
    prediction = np.asarray([row["predicted_utility"] for row in rows])
    utility = np.asarray([row["future_utility"] for row in rows])
    top = np.argsort(-appearance, kind="stable")[:min(top_k, len(rows))]

    def select(indices: np.ndarray, score: np.ndarray, *, gate: bool) -> tuple[float, str | None, bool]:
        local = int(np.argmax(score[indices]))
        choice = int(indices[local])
        accepted = bool(score[choice] > threshold) if gate else True
        return (
            float(utility[choice]) if accepted else 0.0,
            bank[choice]["context_id"] if accepted else None,
            accepted,
        )

    router_topk, selected_context, accepted = select(top, prediction, gate=True)
    router_all, _, accepted_all = select(np.arange(len(rows)), prediction, gate=True)
    appearance_top1, _, _ = select(np.arange(len(rows)), appearance, gate=False)
    oracle_choice = int(np.argmax(utility))
    oracle_all = max(0.0, float(utility[oracle_choice]))
    oracle_topk = max(0.0, float(utility[top].max()))
    return {
        "router_topk": router_topk,
        "router_all": router_all,
        "appearance_top1": appearance_top1,
        "random_expectation": float(utility.mean()),
        "oracle_topk": oracle_topk,
        "oracle_all": oracle_all,
        "router_accepted": accepted,
        "router_all_accepted": accepted_all,
        "selected_context": selected_context,
        "oracle_context": bank[oracle_choice]["context_id"],
        "oracle_in_topk": bank[oracle_choice]["context_id"] in {
            bank[int(index)]["context_id"] for index in top
        },
        "bank_records": len(bank),
        "bank_unique_contexts": len({item["context_id"] for item in bank}),
        "appearance_comparisons": len(bank),
        "router_comparisons": len(top),
    }


def _summary(rows: list[dict], epsilon: float, group_by_episode: dict[str, str]) -> dict:
    metrics = {}
    for method in (
        "router_topk", "router_all", "appearance_top1", "random_expectation",
        "oracle_topk", "oracle_all",
    ):
        values = np.asarray([row[method] for row in rows], dtype=np.float64)
        group_order_values = {}
        for order in sorted({row["order"] for row in rows}):
            for group in sorted(set(group_by_episode.values())):
                subset = [
                    row[method] for row in rows
                    if row["order"] == order and group_by_episode[row["episode"]] == group
                ]
                if subset:
                    group_order_values[f"{order}:{group}"] = float(np.mean(subset))
        group_array = np.asarray(list(group_order_values.values()))
        metrics[method] = {
            "mean_utility": float(values.mean()),
            "median_utility": float(np.median(values)),
            "beneficial_rate": float(np.mean(values > epsilon)),
            "harmful_rate": float(np.mean(values < -epsilon)),
            "raw_sign_harm_rate": float(np.mean(values < 0)),
            "component_order_harm_rate": float(np.mean(group_array < -epsilon)),
            "mean_regret_to_all_oracle": float(np.mean([
                row["oracle_all"] - row[method] for row in rows
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_v10.yaml")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["stage0"]["simulation_result"])
    if output.exists():
        raise RuntimeError(f"EXP-007 causal simulation already exists: {output}")
    table_path = Path(config["stage0"]["utility_table"])
    table = json.loads(table_path.read_text())
    if not (
        table.get("experiment") == "EXP-007"
        and table.get("split") == "train"
        and table.get("validation_accessed") is False
        and table.get("test_accessed") is False
        and table.get("query_or_future_router_input") is False
    ):
        raise RuntimeError("causal simulator requires the train-only leakage-safe utility table")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {record["episode_id"]: groups[index] for index, record in enumerate(records)}
    contexts = {row["context_id"]: row for row in table["contexts"]}
    events = table["events"]
    lookup = {(row["episode"], row["context_id"]): row for row in table["pairs"]}
    if len(lookup) != table["episodes"] * table["unique_contexts"]:
        raise RuntimeError("utility table does not contain every episode/context pair")
    orders = _orders(config["stage0"]["stream_orders"], events)
    top_k = int(config["stage0"]["top_k"])
    threshold = float(config["router"]["utility_threshold"])

    variants: list[tuple[str, int | None]] = [
        ("unbounded_all_write", None), ("unbounded_unique_context", None),
    ] + [
        (policy, int(capacity))
        for policy in config["stage0"]["policies"]
        for capacity in config["stage0"]["capacities"]
    ]
    variant_results = []
    all_rows = []
    for variant_index, (policy, capacity) in enumerate(variants):
        rows = []
        aggregate_counts = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
        final_sizes = []
        for order_index, (order_name, sequence) in enumerate(orders.items()):
            bank: list[dict] = []
            counters = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
            seen_writes = [0]
            rng = random.Random(int(config["seed"]) + 1009 * variant_index + order_index)
            for stream_position, event_index in enumerate(sequence):
                event = events[event_index]
                for context_id in event["pre_query_writes"]:
                    _add(
                        bank, contexts[context_id], policy=policy, capacity=capacity,
                        rng=rng, counters=counters, seen_writes=seen_writes,
                    )
                evaluation = _evaluate(bank, event["episode"], lookup, top_k, threshold)
                rows.append({
                    "policy": policy, "capacity": capacity, "order": order_name,
                    "stream_position": stream_position, "episode": event["episode"], **evaluation,
                })
                for context_id in event["post_query_writes"]:
                    _add(
                        bank, contexts[context_id], policy=policy, capacity=capacity,
                        rng=rng, counters=counters, seen_writes=seen_writes,
                    )
            final_sizes.append(len(bank))
            for key in aggregate_counts:
                aggregate_counts[key] += counters[key]
        summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variant_results.append({
            "policy": policy, "capacity": capacity, "summary": summary,
            "counts": aggregate_counts, "mean_final_records": float(np.mean(final_sizes)),
        })
        all_rows.extend(rows)
        print(json.dumps({
            "policy": policy, "capacity": capacity,
            "router_topk": summary["metrics"]["router_topk"],
            "max_bank": summary["max_bank_records"],
        }), flush=True)

    unbounded = next(row for row in variant_results if row["policy"] == "unbounded_all_write")
    reference = unbounded["summary"]["metrics"]["router_topk"]
    retention_target = float(config["stage0"]["retention_target"])
    maximum_capacity = int(config["stage0"]["maximum_success_capacity"])
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
        success = (
            variant["capacity"] <= maximum_capacity
            and retention >= retention_target
            and metric["harmful_rate"] <= reference["harmful_rate"]
        )
        variant["registered_promising_policy"] = bool(success)
        if success:
            promising.append({"policy": variant["policy"], "capacity": variant["capacity"]})

    atom_bytes = 8 * 196 * (3 + 1 + 64 + 8 + 1) * 4
    result = {
        "experiment": "EXP-007",
        "stage": "stage0_causal_capacity_simulation",
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "offline_utility_label_only",
        "pseudo_stream_not_real_temporal_claim": True,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "utility_table": str(table_path),
        "utility_table_sha256": _sha256(table_path),
        "orders": {name: [events[index]["episode"] for index in order] for name, order in orders.items()},
        "top_k": top_k,
        "estimated_atom_bytes": atom_bytes,
        "variants": variant_results,
        "registered_gate": {
            "retention_target": retention_target,
            "maximum_capacity": maximum_capacity,
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
