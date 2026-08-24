#!/usr/bin/env python3
"""Deployable OOF-calibrated visual buckets for EXP-007 consolidation."""

from __future__ import annotations

import argparse
import itertools
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
    return {
        "visual_bucket_predicted_history": "predicted_history",
        "visual_bucket_delayed_topk_utility": "delayed_topk_utility",
    }[policy]


def _pair_data(streams: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    similarity, labels = [], []
    for stream in streams:
        contexts = stream["contexts"]
        descriptor = np.asarray([row["descriptor"] for row in contexts], dtype=np.float64)
        descriptor /= np.linalg.norm(descriptor, axis=1, keepdims=True).clip(1e-12)
        for left, right in itertools.combinations(range(len(contexts)), 2):
            similarity.append(float(descriptor[left] @ descriptor[right]))
            labels.append(contexts[left]["scene"] == contexts[right]["scene"])
    return np.asarray(similarity), np.asarray(labels, dtype=bool)


def _fit_threshold(streams: list[dict]) -> tuple[float, dict]:
    score, label = _pair_data(streams)
    if not label.any() or label.all():
        raise RuntimeError("visual-bucket threshold requires positive and negative train pairs")
    thresholds = np.unique(np.quantile(score, np.linspace(0, 1, 1001)))
    best = None
    for threshold in thresholds:
        prediction = score >= threshold
        tpr = float(prediction[label].mean())
        tnr = float((~prediction[~label]).mean())
        balanced = 0.5 * (tpr + tnr)
        precision = float(label[prediction].mean()) if prediction.any() else 1.0
        candidate = (balanced, precision, float(threshold), tpr, tnr)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    balanced, precision, threshold, tpr, tnr = best
    return threshold, {
        "pairs": len(score), "positive_pairs": int(label.sum()), "negative_pairs": int((~label).sum()),
        "balanced_accuracy": balanced, "precision": precision,
        "same_scene_recall": tpr, "different_scene_rejection": tnr,
    }


def _evaluate_threshold(stream: dict, threshold: float) -> dict:
    score, label = _pair_data([stream])
    prediction = score >= threshold
    return {
        "pairs": len(score), "positive_pairs": int(label.sum()), "negative_pairs": int((~label).sum()),
        "balanced_accuracy": float(0.5 * (
            prediction[label].mean() + (~prediction[~label]).mean()
        )),
        "precision": float(label[prediction].mean()) if prediction.any() else 1.0,
        "same_scene_recall": float(prediction[label].mean()),
        "different_scene_rejection": float((~prediction[~label]).mean()),
    }


def _write_visual(
    bank: list[dict], context: dict, policy: str, capacity: int, serial: int,
    threshold: float, config: dict, counts: dict,
) -> None:
    counts["writes"] += 1
    descriptor = np.asarray(context["descriptor"], dtype=np.float64)
    descriptor /= np.linalg.norm(descriptor).clip(1e-12)
    merge_index = None
    if bank:
        bank_descriptor = np.asarray([entry["descriptor"] for entry in bank], dtype=np.float64)
        bank_descriptor /= np.linalg.norm(bank_descriptor, axis=1, keepdims=True).clip(1e-12)
        similarity = bank_descriptor @ descriptor
        best = int(np.argmax(similarity))
        if similarity[best] >= threshold:
            merge_index = best
    if merge_index is not None:
        entry = bank[merge_index]
        if context["scene"] in entry["diagnostic_scenes"]:
            counts["true_bucket_merges"] += 1
        else:
            counts["false_bucket_merges"] += 1
        entry.update({
            "context_id": context["context_id"],
            "descriptor": context["descriptor"],
            "serial": serial,
            "frequency": entry["frequency"] + 1,
            "diagnostic_scenes": sorted(set(entry["diagnostic_scenes"] + [context["scene"]])),
        })
        counts["merges"] += 1
        return
    entry = {
        "context_id": context["context_id"],
        "scene": context["scene"],  # diagnostics only; never read by the policy
        "diagnostic_scenes": [context["scene"]],
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
    parser.add_argument("--config", default="configs/EXP-007_visual_bucket_v17.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["bank"]["result"])
    if output.exists():
        raise RuntimeError(f"visual-bucket result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity_result = json.loads(capacity_path.read_text())
    oracle_scene_path = Path(config["source"]["oracle_scene_result"])
    oracle_scene = json.loads(oracle_scene_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and capacity_result.get("validation_accessed") is False
        and oracle_scene.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("v1.7 requires train-only fold-local sources")
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
    threshold_rows = []
    thresholds = {}
    for stream in table["streams"]:
        fold = int(stream["fold"])
        train_streams = [other for other in table["streams"] if int(other["fold"]) != fold]
        threshold, train_metrics = _fit_threshold(train_streams)
        thresholds[fold] = threshold
        threshold_rows.append({
            "held_out_fold": fold, "threshold": threshold,
            "train": train_metrics, "held_out": _evaluate_threshold(stream, threshold),
        })

    capacity = int(config["bank"]["capacity"])
    top_k = int(config["bank"]["top_k"])
    variants = []
    all_rows = []
    for policy in config["bank"]["policies"]:
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
                        _write_visual(
                            bank, contexts[context_id], policy, capacity, serial,
                            thresholds[fold], config["bank"], counts,
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
                        _write_visual(
                            bank, contexts[context_id], policy, capacity, serial,
                            thresholds[fold], config["bank"], counts,
                        )
        summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variants.append({"policy": policy, "capacity": capacity, "summary": summary, "counts": counts})
        all_rows.extend(rows)
        print(json.dumps({"policy": policy, "summary": summary, "counts": counts}), flush=True)

    appearance_variant = next(
        row for row in capacity_result["variants"]
        if row["policy"] == "appearance_diversity" and row["capacity"] == capacity
    )
    appearance = appearance_variant["summary"]["metrics"]["router_topk"]
    appearance_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "appearance_diversity" and row["capacity"] == capacity
    ]
    oracle_variant = next(
        row for row in oracle_scene["variants"]
        if row["policy"] == "scene_delayed_topk_utility"
    )
    oracle_metric = oracle_variant["summary"]
    groups = sorted(set(group_list))
    appearance_component = {
        group: float(np.mean([
            row["router_topk"] for row in appearance_rows if group_by_episode[row["episode"]] == group
        ])) for group in groups
    }
    for variant in variants:
        method_rows = [row for row in all_rows if row["policy"] == variant["policy"]]
        method_component = {
            group: float(np.mean([
                row["router_topk"] for row in method_rows if group_by_episode[row["episode"]] == group
            ])) for group in groups
        }
        variant["minus_appearance_component_bootstrap"] = _bootstrap(
            method_component, appearance_component, groups,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["bootstrap_seed"]),
        )
        metric = variant["summary"]
        variant["oracle_scene_utility_retention"] = metric["mean_utility"] / oracle_metric["mean_utility"]
        variant["registered_pass"] = (
            metric["mean_utility"] > appearance["mean_utility"]
            and metric["harmful_rate"] <= appearance["harmful_rate"]
            and variant["oracle_scene_utility_retention"]
            >= float(config["success"]["minimum_oracle_scene_utility_retention"])
        )
    result = {
        "experiment": "EXP-007",
        "stage": "stage5_deployable_visual_bucket_consolidation",
        "split": "train",
        "protocol_revision": "v1.7",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "same_event_future_in_features": False,
        "ground_truth_scene_runtime_input": False,
        "ground_truth_scene_role": "crossfit_threshold_target_and_diagnostic_only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "thresholds": threshold_rows,
        "appearance_diversity_capacity8": appearance,
        "oracle_scene_delayed_topk": oracle_metric,
        "variants": variants,
        "registered_gate": {
            "passed": any(row["registered_pass"] for row in variants),
            "passing_policies": [row["policy"] for row in variants if row["registered_pass"]],
        },
        "rows": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "thresholds": threshold_rows,
        "appearance": appearance, "oracle_scene": oracle_metric,
        "gate": result["registered_gate"],
        "variants": [{
            "policy": row["policy"], "summary": row["summary"],
            "retention": row["oracle_scene_utility_retention"],
            "difference": row["minus_appearance_component_bootstrap"],
        } for row in variants],
    }), flush=True)


if __name__ == "__main__":
    main()
