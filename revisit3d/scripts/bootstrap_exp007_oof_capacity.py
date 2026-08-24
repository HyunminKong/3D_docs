#!/usr/bin/env python3
"""Paired physical-component bootstrap for EXP-007 OOF bank policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split


def _interval(values: np.ndarray) -> dict:
    low, high = np.percentile(values, [2.5, 97.5])
    return {"bootstrap_mean": float(values.mean()), "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_oof_v12.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    result_path = Path(config["stage0"]["simulation_result"])
    result = json.loads(result_path.read_text())
    if not (
        result.get("protocol_revision") == "v1.2"
        and result.get("validation_accessed") is False
        and result.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("bootstrap requires leakage-safe EXP-007 v1.2 rows")
    output = Path(config["stage0"]["bootstrap_result"])
    if output.exists():
        raise RuntimeError(f"EXP-007 bootstrap already exists: {output}")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, groups = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {record["episode_id"]: groups[index] for index, record in enumerate(records)}
    variants = {
        f"{row['policy']}:{row['capacity']}": row for row in result["variants"]
    }
    row_lookup = {}
    for row in result["rows"]:
        key = f"{row['policy']}:{row['capacity']}"
        row_lookup[(key, row["fold"], row["order"], row["episode"])] = row
    groups_unique = sorted(set(groups))
    orders = sorted({row["order"] for row in result["rows"]})

    # Average pseudo-orders inside each physical component. Components, not
    # order repetitions, are the independent bootstrap units.
    component_value = {}
    for variant_key in variants:
        for group in groups_unique:
            episodes = [episode for episode, value in group_by_episode.items() if value == group]
            selected = []
            oracle = []
            for order in orders:
                matching = [
                    row for (key, _fold, row_order, episode), row in row_lookup.items()
                    if key == variant_key and row_order == order and episode in episodes
                ]
                if matching:
                    selected.extend(row["router_topk"] for row in matching)
                    oracle.extend(row["causal_unbounded_oracle"] for row in matching)
            if not selected:
                raise RuntimeError(f"missing component rows for {variant_key}/{group}")
            component_value[(variant_key, group)] = {
                "router_topk": float(np.mean(selected)),
                "regret": float(np.mean(np.asarray(oracle) - np.asarray(selected))),
            }

    reference = "unbounded_all_write:None"
    compare = [
        reference,
        "unbounded_unique_context:None",
        "fifo:4", "fifo:8",
        "reservoir:4", "reservoir:8",
        "scene_latest:4", "scene_latest:8", "scene_latest:16",
        "appearance_diversity:8", "appearance_diversity:16",
    ]
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    samples = int(config["statistics"]["bootstrap_samples"])
    draws = {key: [] for key in compare}
    differences = {f"{key}_minus_unbounded": [] for key in compare if key != reference}
    for _ in range(samples):
        sampled = rng.choice(groups_unique, size=len(groups_unique), replace=True)
        statistic = {
            key: float(np.mean([component_value[(key, str(group))]["router_topk"] for group in sampled]))
            for key in compare
        }
        for key, value in statistic.items():
            draws[key].append(value)
        for key in compare:
            if key != reference:
                differences[f"{key}_minus_unbounded"].append(statistic[key] - statistic[reference])

    selected_variant = variants["scene_latest:8"]
    unbounded_variant = variants[reference]
    output_payload = {
        "experiment": "EXP-007",
        "stage": "stage0_oof_capacity_component_bootstrap",
        "split": "train",
        "protocol_revision": "v1.2",
        "validation_accessed": False,
        "test_accessed": False,
        "unit": "physical_overlap_component_with_orders_averaged_within_unit",
        "components": len(groups_unique),
        "pseudo_orders": len(orders),
        "samples": samples,
        "seed": int(config["statistics"]["bootstrap_seed"]),
        "source": str(result_path),
        "method_intervals": {key: _interval(np.asarray(value)) for key, value in draws.items()},
        "difference_intervals": {
            key: _interval(np.asarray(value)) for key, value in differences.items()
        },
        "selected_train_policy": {
            "policy": "scene_latest",
            "capacity": 8,
            "selection_reason": "highest OOF mean utility with lower harm than unbounded top-5",
            "mean_utility": selected_variant["summary"]["metrics"]["router_topk"]["mean_utility"],
            "harmful_rate": selected_variant["summary"]["metrics"]["router_topk"]["harmful_rate"],
            "mean_bank_records": selected_variant["summary"]["mean_bank_records"],
            "max_bank_records": selected_variant["summary"]["max_bank_records"],
            "appearance_comparisons": selected_variant["summary"]["appearance_comparisons"],
            "router_comparisons": selected_variant["summary"]["router_comparisons"],
        },
        "unbounded_reference": {
            "mean_utility": unbounded_variant["summary"]["metrics"]["router_topk"]["mean_utility"],
            "harmful_rate": unbounded_variant["summary"]["metrics"]["router_topk"]["harmful_rate"],
            "mean_bank_records": unbounded_variant["summary"]["mean_bank_records"],
            "max_bank_records": unbounded_variant["summary"]["max_bank_records"],
            "appearance_comparisons": unbounded_variant["summary"]["appearance_comparisons"],
            "router_comparisons": unbounded_variant["summary"]["router_comparisons"],
        },
        "component_values": {
            key: {group: component_value[(key, group)] for group in groups_unique}
            for key in compare
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output),
        "selected": output_payload["selected_train_policy"],
        "difference": output_payload["difference_intervals"]["scene_latest:8_minus_unbounded"],
    }))


if __name__ == "__main__":
    main()
