#!/usr/bin/env python3
"""Diagnose candidate-prefilter scaling before EXP-007 consolidation learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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


def _stream_orders(names: list[str], events: list[dict]) -> dict[str, list[int]]:
    base = list(range(len(events)))
    lexical = sorted(base, key=lambda index: events[index]["episode"])
    result = {}
    for name in names:
        if name == "lexicographic":
            result[name] = lexical
        elif name == "reverse_lexicographic":
            result[name] = list(reversed(lexical))
        else:
            result[name] = list(base)
            random.Random(int(name.removeprefix("random_"))).shuffle(result[name])
    return result


def _summarize(rows: list[dict], epsilon: float, group_by_episode: dict[str, str]) -> dict:
    values = np.asarray([row["selected_utility"] for row in rows])
    group_order = []
    for order in sorted({row["order"] for row in rows}):
        for group in sorted(set(group_by_episode.values())):
            subset = [
                row["selected_utility"] for row in rows
                if row["order"] == order and group_by_episode[row["episode"]] == group
            ]
            if subset:
                group_order.append(float(np.mean(subset)))
    return {
        "events": len(rows),
        "mean_utility": float(values.mean()),
        "median_utility": float(np.median(values)),
        "beneficial_rate": float(np.mean(values > epsilon)),
        "harmful_rate": float(np.mean(values < -epsilon)),
        "raw_sign_harm_rate": float(np.mean(values < 0)),
        "component_order_harm_rate": float(np.mean(np.asarray(group_order) < -epsilon)),
        "mean_regret_to_causal_all_oracle": float(np.mean([row["regret"] for row in rows])),
        "oracle_recall": float(np.mean([row["oracle_recalled"] for row in rows])),
        "mean_bank_records": float(np.mean([row["bank_records"] for row in rows])),
        "mean_prefilter_comparisons": float(np.mean([row["bank_records"] for row in rows])),
        "mean_full_router_comparisons": float(np.mean([row["rerank_count"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["stage0"]["prefilter_result"])
    if output.exists():
        raise RuntimeError(f"prefilter scaling result already exists: {output}")
    table_path = Path(config["stage0"]["utility_table"])
    table = json.loads(table_path.read_text())
    if table.get("validation_accessed") is not False or table.get("split") != "train":
        raise RuntimeError("prefilter analysis requires the train-only table")
    contexts = {row["context_id"]: row for row in table["contexts"]}
    events = table["events"]
    lookup = {(row["episode"], row["context_id"]): row for row in table["pairs"]}
    orders = _stream_orders(config["stage0"]["stream_orders"], events)
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {record["episode_id"]: groups[index] for index, record in enumerate(records)}
    epsilon = float(table["utility_deadband"])
    threshold = float(config["router"]["utility_threshold"])
    requested_k = config["stage0"]["prefilter_k_curve"]

    method_rows: dict[str, list[dict]] = {}
    for bank_mode in ("all_write", "unique_context"):
        for order_name, sequence in orders.items():
            bank: list[str] = []
            for event_index in sequence:
                event = events[event_index]
                for context_id in event["pre_query_writes"]:
                    if bank_mode == "unique_context" and context_id in bank:
                        continue
                    bank.append(context_id)
                pair_rows = [lookup[(event["episode"], context_id)] for context_id in bank]
                utility = np.asarray([row["future_utility"] for row in pair_rows])
                appearance = np.asarray([row["appearance_similarity"] for row in pair_rows])
                current = np.asarray([row["current_objective_improvement"] for row in pair_rows])
                router = np.asarray([row["predicted_utility"] for row in pair_rows])
                oracle_index = int(np.argmax(utility))
                oracle_value = max(0.0, float(utility[oracle_index]))
                for value in requested_k:
                    k = len(bank) if value == "all" else min(int(value), len(bank))
                    for prefilter_name, prefilter_score in (
                        ("appearance", appearance),
                        ("current_objective", current),
                        ("router_score_upper_bound", router),
                    ):
                        candidates = np.argsort(-prefilter_score, kind="stable")[:k]
                        choice = int(candidates[np.argmax(router[candidates])])
                        accepted = bool(router[choice] > threshold)
                        selected = float(utility[choice]) if accepted else 0.0
                        name = f"{bank_mode}:{prefilter_name}:k={value}"
                        method_rows.setdefault(name, []).append({
                            "order": order_name,
                            "episode": event["episode"],
                            "selected_utility": selected,
                            "accepted": accepted,
                            "regret": oracle_value - selected,
                            "oracle_recalled": oracle_index in set(candidates.tolist()),
                            "bank_records": len(bank),
                            "rerank_count": len(candidates),
                        })
                for context_id in event["post_query_writes"]:
                    if bank_mode == "unique_context" and context_id in bank:
                        continue
                    bank.append(context_id)

    summaries = {
        name: _summarize(rows, epsilon, group_by_episode) for name, rows in method_rows.items()
    }
    selected = {
        name: summary for name, summary in summaries.items()
        if name.startswith("unique_context:appearance")
        or name in (
            "unique_context:router_score_upper_bound:k=all",
            "unique_context:current_objective:k=5",
            "unique_context:current_objective:k=all",
        )
    }
    result = {
        "experiment": "EXP-007",
        "stage": "stage0_prefilter_scaling_diagnostic",
        "split": "train",
        "protocol_revision": "v1.1",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "offline_utility_label_only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "utility_table": str(table_path),
        "utility_table_sha256": _sha256(table_path),
        "k_curve": requested_k,
        "interpretation_contract": {
            "appearance": "deployable cheap prefilter",
            "current_objective": "transport-every-candidate diagnostic, not scalable",
            "router_score_upper_bound": "full observable score over every candidate; compute upper bound",
        },
        "selected_summaries": selected,
        "all_summaries": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
