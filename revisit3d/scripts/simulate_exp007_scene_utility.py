#!/usr/bin/env python3
"""Scene-bucketed, utility-aware consolidation for EXP-007 v1.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _evaluate, _orders, _sha256
from revisit3d.scripts.simulate_exp007_utility_consolidation import (
    _score,
    _summary,
    _update_history,
)


def _base_policy(policy: str) -> str:
    mapping = {
        "scene_predicted_history": "predicted_history",
        "scene_selected_utility_ucb": "selected_utility_ucb",
        "scene_delayed_topk_utility": "delayed_topk_utility",
        "scene_hybrid_history": "hybrid_history",
    }
    return mapping[policy]


def _write_scene(
    bank: list[dict], context: dict, policy: str, capacity: int, serial: int,
    config: dict, counts: dict,
) -> None:
    counts["writes"] += 1
    existing = next((entry for entry in bank if entry["scene"] == context["scene"]), None)
    if existing is not None:
        existing.update({
            "context_id": context["context_id"],
            "descriptor": context["descriptor"],
            "serial": serial,
            "frequency": existing["frequency"] + 1,
        })
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
    if len(candidates) > capacity:
        base = _base_policy(policy)
        remove = min(
            range(len(candidates)),
            key=lambda index: (_score(candidates[index], base, config), candidates[index]["serial"]),
        )
        if remove == len(bank):
            counts["rejected_writes"] += 1
        else:
            counts["evictions"] += 1
        candidates.pop(remove)
    bank[:] = candidates


def _bootstrap(
    left: dict[str, float], right: dict[str, float], groups: list[str], samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draws.append(float(np.mean([left[str(group)] - right[str(group)] for group in sampled])))
    array = np.asarray(draws)
    low, high = np.percentile(array, [2.5, 97.5])
    return {"bootstrap_mean": float(array.mean()), "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_scene_utility_v16.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["bank"]["result"])
    if output.exists():
        raise RuntimeError(f"scene-utility result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity_result = json.loads(capacity_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and capacity_result.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("v1.6 requires train-only fold-local sources")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, group_list = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {
        record["episode_id"]: group_list[index] for index, record in enumerate(records)
    }
    reference_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "unbounded_all_write" and row["capacity"] is None
    ]
    oracle_reference = {
        (row["fold"], row["order"], row["episode"]): row["causal_unbounded_oracle"]
        for row in reference_rows
    }
    capacity = int(config["bank"]["capacity"])
    top_k = int(config["bank"]["top_k"])
    variants = []
    all_rows = []
    for policy in config["bank"]["policies"]:
        rows = []
        counts = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
        for stream in table["streams"]:
            contexts = {row["context_id"]: row for row in stream["contexts"]}
            lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
            orders = _orders(config["bank"]["stream_orders"], stream["events"])
            for order_index, (order_name, sequence) in enumerate(orders.items()):
                bank = []
                serial = int(config["seed"]) + 100000 * int(stream["fold"]) + 1000 * order_index
                for position, event_index in enumerate(sequence):
                    event = stream["events"][event_index]
                    for context_id in event["pre_query_writes"]:
                        serial += 1
                        _write_scene(
                            bank, contexts[context_id], policy, capacity, serial,
                            config["bank"], counts,
                        )
                    evaluation = _evaluate(bank, event["episode"], lookup, top_k, 0.0)
                    evaluation["causal_unbounded_oracle"] = oracle_reference[
                        (stream["fold"], order_name, event["episode"])
                    ]
                    rows.append({
                        "fold": stream["fold"], "policy": policy, "capacity": capacity,
                        "order": order_name, "stream_position": position,
                        "episode": event["episode"], **evaluation,
                    })
                    _update_history(
                        bank, event["episode"], lookup, _base_policy(policy), top_k,
                        evaluation["selected_context"],
                    )
                    for context_id in event["post_query_writes"]:
                        serial += 1
                        _write_scene(
                            bank, contexts[context_id], policy, capacity, serial,
                            config["bank"], counts,
                        )
        summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variants.append({"policy": policy, "capacity": capacity, "summary": summary, "counts": counts})
        all_rows.extend(rows)
        print(json.dumps({"policy": policy, "summary": summary}), flush=True)

    scene_variant = next(
        row for row in capacity_result["variants"]
        if row["policy"] == "scene_latest" and row["capacity"] == capacity
    )
    scene = scene_variant["summary"]["metrics"]["router_topk"]
    scene_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "scene_latest" and row["capacity"] == capacity
    ]
    groups = sorted(set(group_list))
    scene_component = {
        group: float(np.mean([
            row["router_topk"] for row in scene_rows if group_by_episode[row["episode"]] == group
        ])) for group in groups
    }
    for variant in variants:
        method_rows = [row for row in all_rows if row["policy"] == variant["policy"]]
        method_component = {
            group: float(np.mean([
                row["router_topk"] for row in method_rows if group_by_episode[row["episode"]] == group
            ])) for group in groups
        }
        variant["minus_scene_component_bootstrap"] = _bootstrap(
            method_component, scene_component, groups,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["bootstrap_seed"]),
        )
        metric = variant["summary"]
        variant["beats_scene_latest"] = (
            metric["mean_utility"] > scene["mean_utility"]
            and metric["harmful_rate"] <= scene["harmful_rate"]
        )
    result = {
        "experiment": "EXP-007",
        "stage": "stage4_scene_bucket_utility_consolidation",
        "split": "train",
        "protocol_revision": "v1.6",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "same_event_future_in_features": False,
        "past_delayed_utility_history": True,
        "scene_role": "coarse_redundancy_bucket_only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "scene_latest_capacity8": scene,
        "variants": variants,
        "registered_gate": {
            "passed": any(row["beats_scene_latest"] for row in variants),
            "passing_policies": [row["policy"] for row in variants if row["beats_scene_latest"]],
        },
        "rows": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "scene": scene,
        "gate": result["registered_gate"],
        "variants": [{
            "policy": row["policy"],
            "utility": row["summary"]["mean_utility"],
            "harm": row["summary"]["harmful_rate"],
            "difference": row["minus_scene_component_bootstrap"],
        } for row in variants],
    }), flush=True)


if __name__ == "__main__":
    main()
