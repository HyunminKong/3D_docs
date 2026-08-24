#!/usr/bin/env python3
"""Set-normalized OOF reranking for persistent EXP-007 candidate lists."""

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

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _orders, _sha256
from revisit3d.scripts.simulate_exp007_utility_consolidation import (
    _score,
    _update_history,
    _write,
)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    rank = np.empty(len(values), dtype=np.float64)
    rank[order] = np.arange(len(values), dtype=np.float64)
    return rank / max(len(values) - 1, 1)


def _metrics(
    events: list[dict], scores: np.ndarray, threshold: float | None,
    epsilon: float, group_by_episode: dict[str, str],
) -> tuple[dict, dict[str, float]]:
    values, accepted, regret = [], [], []
    component_values: dict[str, list[float]] = {}
    for event in events:
        indices = np.asarray(event["candidate_indices"])
        choice = int(indices[np.argmax(scores[indices])])
        take = True if threshold is None else bool(scores[choice] > threshold)
        value = float(event["targets"][event["candidate_indices"].index(choice)]) if take else 0.0
        oracle = max(0.0, max(event["targets"]))
        values.append(value)
        accepted.append(take)
        regret.append(oracle - value)
        group = group_by_episode[event["episode"]]
        component_values.setdefault(group, []).append(value)
    array = np.asarray(values)
    component_mean = {group: float(np.mean(value)) for group, value in component_values.items()}
    return {
        "events": len(events),
        "mean_utility": float(array.mean()),
        "median_utility": float(np.median(array)),
        "beneficial_rate": float(np.mean(array > epsilon)),
        "harmful_rate": float(np.mean(array < -epsilon)),
        "raw_sign_harm_rate": float(np.mean(array < 0)),
        "component_harm_rate": float(np.mean(np.asarray(list(component_mean.values())) < -epsilon)),
        "accept_rate": float(np.mean(accepted)),
        "mean_regret_to_topk_oracle": float(np.mean(regret)),
        "threshold": threshold,
    }, component_mean


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
    parser.add_argument("--config", default="configs/EXP-007_listwise_router_v15.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["router"]["result"])
    if output.exists():
        raise RuntimeError(f"listwise result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity = json.loads(capacity_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and capacity.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("listwise router requires train-only fold-local sources")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, group_list = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {
        record["episode_id"]: group_list[index] for index, record in enumerate(records)
    }
    bank_config = config["bank"]
    bank_capacity = int(bank_config["capacity"])
    top_k = int(bank_config["top_k"])
    candidate_rows = []
    event_rows = []
    serial_seed = int(config["seed"])
    for stream in table["streams"]:
        contexts = {row["context_id"]: row for row in stream["contexts"]}
        lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
        orders = _orders(bank_config["stream_orders"], stream["events"])
        for order_index, (order_name, sequence) in enumerate(orders.items()):
            bank = []
            counts = {"writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0}
            serial = serial_seed + 100000 * int(stream["fold"]) + 1000 * order_index
            for position, event_index in enumerate(sequence):
                event = stream["events"][event_index]
                future = [stream["events"][index]["episode"] for index in sequence[position:]]
                for context_id in event["pre_query_writes"]:
                    serial += 1
                    _write(
                        bank, contexts[context_id], bank_config["policy"], bank_capacity,
                        serial, bank_config, future, lookup, counts,
                    )
                rows = [lookup[(event["episode"], entry["context_id"])] for entry in bank]
                appearance_all = np.asarray([row["appearance_similarity"] for row in rows])
                top = np.argsort(-appearance_all, kind="stable")[:min(top_k, len(bank))]
                raw_prediction = np.asarray([rows[int(index)]["predicted_utility"] for index in top])
                appearance = appearance_all[top]
                current = np.asarray([
                    rows[int(index)]["current_objective_improvement"] for index in top
                ])
                raw_stack = np.stack((raw_prediction, appearance, current), axis=1)
                normalized = (raw_stack - raw_stack.mean(axis=0, keepdims=True)) / raw_stack.std(
                    axis=0, keepdims=True,
                ).clip(1e-6)
                ranks = np.stack(tuple(_ranks(raw_stack[:, column]) for column in range(3)), axis=1)
                indices = []
                targets = []
                for local, bank_index in enumerate(top):
                    entry = bank[int(bank_index)]
                    pred_mean = (
                        entry["pred_sum"] + float(bank_config["prior_count"]) * float(bank_config["prior_utility"])
                    ) / (entry["pred_count"] + float(bank_config["prior_count"]))
                    utility_mean = (
                        entry["utility_sum"] + float(bank_config["prior_count"]) * float(bank_config["prior_utility"])
                    ) / (entry["utility_count"] + float(bank_config["prior_count"]))
                    feature = [
                        *[float(value) for value in raw_stack[local]],
                        *[float(value) for value in normalized[local]],
                        *[float(value) for value in ranks[local]],
                        len(bank) / bank_capacity,
                        float(pred_mean),
                        float(np.log1p(entry["pred_count"])),
                        float(utility_mean),
                        float(np.log1p(entry["utility_count"])),
                        float(_score(entry, bank_config["policy"], bank_config)),
                        float(np.log1p(entry["frequency"])),
                        float((serial - entry["serial"]) / max(serial, 1)),
                    ]
                    target = float(rows[int(bank_index)]["future_utility"])
                    indices.append(len(candidate_rows))
                    targets.append(target)
                    candidate_rows.append({
                        "fold": stream["fold"],
                        "order": order_name,
                        "episode": event["episode"],
                        "group": group_by_episode[event["episode"]],
                        "context_id": entry["context_id"],
                        "features": feature,
                        "future_utility": target,
                    })
                event_rows.append({
                    "fold": stream["fold"],
                    "order": order_name,
                    "episode": event["episode"],
                    "group": group_by_episode[event["episode"]],
                    "candidate_indices": indices,
                    "targets": targets,
                })
                raw_choice = int(top[np.argmax(raw_prediction)])
                _update_history(
                    bank, event["episode"], lookup, bank_config["policy"], top_k,
                    bank[raw_choice]["context_id"],
                )
                future_after = [
                    stream["events"][index]["episode"] for index in sequence[position + 1:]
                ]
                for context_id in event["post_query_writes"]:
                    serial += 1
                    _write(
                        bank, contexts[context_id], bank_config["policy"], bank_capacity,
                        serial, bank_config, future_after, lookup, counts,
                    )

    matrix = np.asarray([row["features"] for row in candidate_rows], dtype=np.float64)
    target = np.asarray([row["future_utility"] for row in candidate_rows])
    groups = np.asarray([row["group"] for row in candidate_rows])
    feature_sets = {
        "set_normalized_pointwise_ridge": np.arange(17),
        "pointwise_without_set_context": np.asarray([0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16]),
        "set_context_without_history": np.arange(10),
        "history_only": np.arange(9, 17),
    }
    predictions = {name: np.empty(len(candidate_rows)) for name in feature_sets}
    pairwise_prediction = np.empty(len(candidate_rows))
    fold_rows = []
    unique_groups = sorted(set(groups.tolist()))
    for held_out in unique_groups:
        train, test = groups != held_out, groups == held_out
        diagnostics = {}
        for name, columns in feature_sets.items():
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=float(config["router"]["ridge_alpha"])),
            )
            model.fit(matrix[train][:, columns], target[train])
            predictions[name][test] = model.predict(matrix[test][:, columns])
            diagnostics[name] = float(np.abs(predictions[name][test] - target[test]).mean())

        difference_x, difference_y = [], []
        for event in event_rows:
            if event["group"] == held_out:
                continue
            indices = [index for index in event["candidate_indices"] if train[index]]
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    i, j = indices[left], indices[right]
                    difference_x.append(matrix[i] - matrix[j])
                    difference_y.append(target[i] - target[j])
        pairwise = make_pipeline(
            StandardScaler(), Ridge(alpha=float(config["router"]["ridge_alpha"])),
        )
        pairwise.fit(np.asarray(difference_x), np.asarray(difference_y))
        pairwise_prediction[test] = pairwise.predict(matrix[test])
        diagnostics["pairwise_ridge"] = None
        fold_rows.append({
            "held_out_component": held_out,
            "train_candidates": int(train.sum()),
            "test_candidates": int(test.sum()),
            "candidate_mae": diagnostics,
        })

    epsilon = float(table["utility_deadband"])
    threshold = float(config["router"]["utility_threshold"])
    metrics = {}
    component_metrics = {}
    for name, score in predictions.items():
        metrics[f"{name}_epsilon"], component_metrics[f"{name}_epsilon"] = _metrics(
            event_rows, score, threshold, epsilon, group_by_episode,
        )
        metrics[f"{name}_zero"], component_metrics[f"{name}_zero"] = _metrics(
            event_rows, score, float(config["router"]["zero_threshold_control"]),
            epsilon, group_by_episode,
        )
    metrics["pairwise_ridge_rank_only"], component_metrics["pairwise_ridge_rank_only"] = _metrics(
        event_rows, pairwise_prediction, None, epsilon, group_by_episode,
    )
    raw_score = matrix[:, 0]
    metrics["raw_router"], component_metrics["raw_router"] = _metrics(
        event_rows, raw_score, 0.0, epsilon, group_by_episode,
    )
    metrics["appearance"], component_metrics["appearance"] = _metrics(
        event_rows, matrix[:, 1], None, epsilon, group_by_episode,
    )
    metrics["current_objective"], component_metrics["current_objective"] = _metrics(
        event_rows, matrix[:, 2], 0.0, epsilon, group_by_episode,
    )
    metrics["oracle_topk"], component_metrics["oracle_topk"] = _metrics(
        event_rows, target, 0.0, epsilon, group_by_episode,
    )

    primary_name = f"{config['router']['primary']}_epsilon"
    primary = metrics[primary_name]
    scene_variant = next(
        row for row in capacity["variants"]
        if row["policy"] == "scene_latest" and row["capacity"] == bank_capacity
    )
    scene = scene_variant["summary"]["metrics"]["router_topk"]
    scene_rows = [
        row for row in capacity["rows"]
        if row["policy"] == "scene_latest" and row["capacity"] == bank_capacity
    ]
    scene_component = {
        group: float(np.mean([
            row["router_topk"] for row in scene_rows if group_by_episode[row["episode"]] == group
        ])) for group in unique_groups
    }
    difference = _bootstrap(
        component_metrics[primary_name], scene_component, unique_groups,
        int(config["statistics"]["bootstrap_samples"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    checks = {
        "beats_scene_latest_mean": primary["mean_utility"] > scene["mean_utility"],
        "harm_not_above_scene_latest": primary["harmful_rate"] <= scene["harmful_rate"],
        "nontrivial_acceptance": primary["accept_rate"] >= float(config["success"]["minimum_accept_rate"]),
    }
    result = {
        "experiment": "EXP-007",
        "stage": "stage3_set_normalized_listwise_router_logo",
        "split": "train",
        "protocol_revision": "v1.5",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "target_and_past_delayed_history_only",
        "same_event_future_in_features": False,
        "off_policy_candidate_sets": True,
        "bank_policy": bank_config["policy"],
        "bank_capacity": bank_capacity,
        "grouping": "leave_one_physical_overlap_component_out",
        "components": len(unique_groups),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "feature_contract": {
            "dimensions": 17,
            "scene_identity_input": False,
            "set_normalized": True,
            "query_or_future_input": False,
        },
        "candidate_label_health": {
            "beneficial": int(np.sum(target > epsilon)),
            "neutral": int(np.sum(np.abs(target) <= epsilon)),
            "harmful": int(np.sum(target < -epsilon)),
        },
        "candidate_utility_spearman": {
            **{name: float(spearmanr(score, target).statistic) for name, score in predictions.items()},
            "pairwise_ridge": float(spearmanr(pairwise_prediction, target).statistic),
            "raw_router": float(spearmanr(raw_score, target).statistic),
        },
        "metrics": metrics,
        "scene_latest_capacity8": scene,
        "primary": primary_name,
        "primary_minus_scene_component_bootstrap": difference,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "folds": fold_rows,
        "events": event_rows,
        "candidates": [{
            **row,
            **{name: float(score[index]) for name, score in predictions.items()},
            "pairwise_ridge": float(pairwise_prediction[index]),
        } for index, row in enumerate(candidate_rows)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "candidate_labels": result["candidate_label_health"],
        "spearman": result["candidate_utility_spearman"],
        "primary": primary, "scene": scene, "difference": difference,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
