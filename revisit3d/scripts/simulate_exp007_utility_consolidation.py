#!/usr/bin/env python3
"""Evaluate causal utility-history consolidation against scene-latest OOF banks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _evaluate, _orders, _sha256


def _score(entry: dict, policy: str, config: dict) -> float:
    prior = float(config["prior_utility"])
    prior_count = float(config["prior_count"])
    bonus = float(config["exploration_bonus"])
    predicted = (entry["pred_sum"] + prior_count * prior) / (entry["pred_count"] + prior_count)
    realized = (entry["utility_sum"] + prior_count * prior) / (entry["utility_count"] + prior_count)
    uncertainty = bonus / np.sqrt(entry["utility_count"] + entry["pred_count"] + 1.0)
    if policy == "predicted_history":
        return float(predicted + uncertainty)
    if policy in ("selected_utility_ucb", "delayed_topk_utility"):
        return float(realized + uncertainty)
    if policy == "hybrid_history":
        return float(0.5 * predicted + 0.5 * realized + uncertainty)
    raise ValueError(f"history score is undefined for {policy!r}")


def _oracle_coverage_subset(
    candidates: list[dict], capacity: int, future_episodes: list[str], lookup: dict,
) -> list[dict]:
    if len(candidates) <= capacity:
        return candidates
    if not future_episodes:
        return sorted(candidates, key=lambda entry: entry["serial"], reverse=True)[:capacity]
    utility = np.asarray([
        [max(0.0, lookup[(episode, entry["context_id"])]["future_utility"])
         for episode in future_episodes]
        for entry in candidates
    ])
    selected = []
    covered = np.zeros(len(future_episodes), dtype=np.float64)
    available = set(range(len(candidates)))
    for _ in range(capacity):
        choice = max(
            available,
            key=lambda index: float(np.maximum(covered, utility[index]).sum() - covered.sum()),
        )
        selected.append(choice)
        covered = np.maximum(covered, utility[choice])
        available.remove(choice)
    return [candidates[index] for index in selected]


def _write(
    bank: list[dict], context: dict, policy: str, capacity: int, serial: int,
    config: dict, future_episodes: list[str], lookup: dict, counts: dict,
) -> None:
    counts["writes"] += 1
    duplicate = next(
        (entry for entry in bank if entry["context_id"] == context["context_id"]), None,
    )
    if duplicate is not None:
        duplicate["frequency"] += 1
        duplicate["serial"] = serial
        counts["merges"] += 1
        return
    entry = {
        "context_id": context["context_id"],
        "scene": context["scene"],
        "descriptor": context["descriptor"],
        "serial": serial,
        "frequency": 1,
        "pred_sum": 0.0,
        "pred_count": 0,
        "utility_sum": 0.0,
        "utility_count": 0,
    }
    candidates = bank + [entry]
    if len(candidates) <= capacity:
        bank[:] = candidates
        return
    if policy == "oracle_future_coverage":
        kept = _oracle_coverage_subset(candidates, capacity, future_episodes, lookup)
        if entry not in kept:
            counts["rejected_writes"] += 1
        else:
            counts["evictions"] += 1
        bank[:] = kept
        return
    remove = min(
        range(len(candidates)),
        key=lambda index: (_score(candidates[index], policy, config), candidates[index]["serial"]),
    )
    if remove == len(bank):
        counts["rejected_writes"] += 1
    else:
        counts["evictions"] += 1
    candidates.pop(remove)
    bank[:] = candidates


def _update_history(
    bank: list[dict], episode: str, lookup: dict, policy: str, top_k: int,
    selected_context: str | None,
) -> None:
    if policy == "oracle_future_coverage":
        return
    rows = [lookup[(episode, entry["context_id"])] for entry in bank]
    appearance = np.asarray([row["appearance_similarity"] for row in rows])
    top = np.argsort(-appearance, kind="stable")[:min(top_k, len(bank))]
    if policy in ("predicted_history", "hybrid_history"):
        for index in top:
            bank[int(index)]["pred_sum"] += float(rows[int(index)]["predicted_utility"])
            bank[int(index)]["pred_count"] += 1
    if policy in ("selected_utility_ucb", "hybrid_history") and selected_context is not None:
        entry = next(item for item in bank if item["context_id"] == selected_context)
        entry["utility_sum"] += float(lookup[(episode, selected_context)]["future_utility"])
        entry["utility_count"] += 1
    if policy == "delayed_topk_utility":
        for index in top:
            entry = bank[int(index)]
            entry["utility_sum"] += float(rows[int(index)]["future_utility"])
            entry["utility_count"] += 1


def _summary(rows: list[dict], epsilon: float, group_by_episode: dict[str, str]) -> dict:
    values = np.asarray([row["router_topk"] for row in rows])
    component_order = []
    for fold, order in sorted({(row["fold"], row["order"]) for row in rows}):
        for group in sorted(set(group_by_episode.values())):
            subset = [
                row["router_topk"] for row in rows
                if row["fold"] == fold and row["order"] == order
                and group_by_episode[row["episode"]] == group
            ]
            if subset:
                component_order.append(float(np.mean(subset)))
    return {
        "events": len(rows),
        "mean_utility": float(values.mean()),
        "median_utility": float(np.median(values)),
        "beneficial_rate": float(np.mean(values > epsilon)),
        "harmful_rate": float(np.mean(values < -epsilon)),
        "raw_sign_harm_rate": float(np.mean(values < 0)),
        "component_order_harm_rate": float(np.mean(np.asarray(component_order) < -epsilon)),
        "mean_regret_to_causal_unbounded_oracle": float(np.mean([
            row["causal_unbounded_oracle"] - row["router_topk"] for row in rows
        ])),
        "router_accept_rate": float(np.mean([row["router_accepted"] for row in rows])),
        "mean_bank_records": float(np.mean([row["bank_records"] for row in rows])),
        "max_bank_records": max(row["bank_records"] for row in rows),
        "oracle_topk_recall": float(np.mean([row["oracle_in_topk"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_utility_consolidation_v13.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["consolidation"]["result"])
    if output.exists():
        raise RuntimeError(f"utility consolidation result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity_result = json.loads(capacity_path.read_text())
    if not (
        table.get("protocol_revision") == capacity_result.get("protocol_revision") == "v1.2"
        and table.get("validation_accessed") is False
        and capacity_result.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("v1.3 requires leakage-safe fold-local v1.2 sources")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {record["episode_id"]: groups[index] for index, record in enumerate(records)}
    reference_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "unbounded_all_write" and row["capacity"] is None
    ]
    oracle_reference = {
        (row["fold"], row["order"], row["episode"]): row["causal_unbounded_oracle"]
        for row in reference_rows
    }
    scene_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "scene_latest" and row["capacity"] == 8
    ]
    scene_summary = _summary(scene_rows, float(table["utility_deadband"]), group_by_episode)
    top_k = int(config["consolidation"]["top_k"])
    variants = []
    all_rows = []
    serial_seed = int(config["seed"])
    for policy in config["consolidation"]["policies"]:
        for capacity in config["consolidation"]["capacities"]:
            rows = []
            counts = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
            for stream in table["streams"]:
                contexts = {row["context_id"]: row for row in stream["contexts"]}
                lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
                orders = _orders(config["consolidation"]["stream_orders"], stream["events"])
                for order_index, (order_name, sequence) in enumerate(orders.items()):
                    bank = []
                    serial = serial_seed + 100000 * int(stream["fold"]) + 1000 * order_index
                    for position, event_index in enumerate(sequence):
                        event = stream["events"][event_index]
                        future = [stream["events"][index]["episode"] for index in sequence[position:]]
                        for context_id in event["pre_query_writes"]:
                            serial += 1
                            _write(
                                bank, contexts[context_id], policy, int(capacity), serial,
                                config["consolidation"], future, lookup, counts,
                            )
                        evaluation = _evaluate(
                            bank, event["episode"], lookup, top_k,
                            float(config["consolidation"]["utility_threshold"]),
                        )
                        evaluation["causal_unbounded_oracle"] = oracle_reference[
                            (stream["fold"], order_name, event["episode"])
                        ]
                        rows.append({
                            "fold": stream["fold"], "policy": policy, "capacity": int(capacity),
                            "order": order_name, "stream_position": position,
                            "episode": event["episode"], **evaluation,
                        })
                        _update_history(
                            bank, event["episode"], lookup, policy, top_k,
                            evaluation["selected_context"],
                        )
                        future_after = [
                            stream["events"][index]["episode"] for index in sequence[position + 1:]
                        ]
                        for context_id in event["post_query_writes"]:
                            serial += 1
                            _write(
                                bank, contexts[context_id], policy, int(capacity), serial,
                                config["consolidation"], future_after, lookup, counts,
                            )
            summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
            variant = {"policy": policy, "capacity": int(capacity), "summary": summary, "counts": counts}
            variants.append(variant)
            all_rows.extend(rows)
            print(json.dumps({"policy": policy, "capacity": capacity, "summary": summary}), flush=True)

    primary_capacity = int(config["success"]["primary_capacity"])
    candidates = [
        row for row in variants
        if row["capacity"] == primary_capacity and row["policy"] != "oracle_future_coverage"
    ]
    for row in candidates:
        row["beats_scene_latest"] = (
            row["summary"]["mean_utility"] > scene_summary["mean_utility"]
            and row["summary"]["harmful_rate"] <= scene_summary["harmful_rate"]
        )
    oracle = next(
        row for row in variants
        if row["capacity"] == primary_capacity and row["policy"] == "oracle_future_coverage"
    )
    result = {
        "experiment": "EXP-007",
        "stage": "stage1_oof_utility_history_consolidation",
        "split": "train",
        "protocol_revision": "v1.3",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "delayed_history_update_or_oracle_control_only",
        "fold_local_memory_coordinates": True,
        "cross_fold_bank_mixing": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "source_table_sha256": _sha256(table_path),
        "scene_latest_capacity8": scene_summary,
        "variants": variants,
        "registered_gate": {
            "primary_capacity": primary_capacity,
            "history_candidates": [
                {"policy": row["policy"], "passed": row["beats_scene_latest"]}
                for row in candidates
            ],
            "passed": any(row["beats_scene_latest"] for row in candidates),
            "oracle_headroom": oracle["summary"]["mean_utility"] - scene_summary["mean_utility"],
        },
        "rows": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "gate": result["registered_gate"]}), flush=True)


if __name__ == "__main__":
    main()
