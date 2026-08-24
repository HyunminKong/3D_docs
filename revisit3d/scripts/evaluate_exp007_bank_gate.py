#!/usr/bin/env python3
"""Leave-one-component-out bank-aware acceptance calibration for EXP-007."""

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


def _gate_metrics(
    rows: list[dict], score: np.ndarray, threshold: float, epsilon: float,
    group_by_episode: dict[str, str],
) -> dict:
    target = np.asarray([row["future_utility"] for row in rows])
    accepted = score > threshold
    value = np.where(accepted, target, 0.0)
    component = []
    for group in sorted(set(group_by_episode.values())):
        subset = [
            value[index] for index, row in enumerate(rows)
            if group_by_episode[row["episode"]] == group
        ]
        component.append(float(np.mean(subset)))
    oracle = np.maximum(target, 0.0)
    return {
        "events": len(rows),
        "mean_utility": float(value.mean()),
        "median_utility": float(np.median(value)),
        "beneficial_rate": float(np.mean(value > epsilon)),
        "harmful_rate": float(np.mean(value < -epsilon)),
        "raw_sign_harm_rate": float(np.mean(value < 0)),
        "component_harm_rate": float(np.mean(np.asarray(component) < -epsilon)),
        "accept_rate": float(accepted.mean()),
        "mean_regret_to_selected_or_reject_oracle": float(np.mean(oracle - value)),
        "threshold": threshold,
    }


def _bootstrap_difference(
    primary: dict[str, float], scene: dict[str, float], groups: list[str], samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draws.append(float(np.mean([primary[str(group)] - scene[str(group)] for group in sampled])))
    array = np.asarray(draws)
    low, high = np.percentile(array, [2.5, 97.5])
    return {"bootstrap_mean": float(array.mean()), "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_bank_gate_v14.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["gate"]["result"])
    if output.exists():
        raise RuntimeError(f"bank-aware gate result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity = json.loads(capacity_path.read_text())
    consolidation_path = Path(config["source"]["consolidation_result"])
    consolidation = json.loads(consolidation_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and capacity.get("validation_accessed") is False
        and consolidation.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("v1.4 requires train-only fold-local OOF sources")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, group_list = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {
        record["episode_id"]: group_list[index] for index, record in enumerate(records)
    }
    bank_config = config["bank"]
    capacity_value = int(bank_config["capacity"])
    top_k = int(bank_config["top_k"])
    feature_rows = []
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
                        bank, contexts[context_id], bank_config["policy"], capacity_value,
                        serial, bank_config, future, lookup, counts,
                    )
                candidates = [lookup[(event["episode"], entry["context_id"])] for entry in bank]
                appearance = np.asarray([row["appearance_similarity"] for row in candidates])
                top = np.argsort(-appearance, kind="stable")[:min(top_k, len(bank))]
                prediction = np.asarray([row["predicted_utility"] for row in candidates])
                choice = int(top[np.argmax(prediction[top])])
                selected = bank[choice]
                top_prediction = np.sort(prediction[top])[::-1]
                second = float(top_prediction[1] if len(top_prediction) > 1 else top_prediction[0])
                selected_pred_mean = (
                    selected["pred_sum"] + float(bank_config["prior_count"]) * float(bank_config["prior_utility"])
                ) / (selected["pred_count"] + float(bank_config["prior_count"]))
                selected_utility_mean = (
                    selected["utility_sum"] + float(bank_config["prior_count"]) * float(bank_config["prior_utility"])
                ) / (selected["utility_count"] + float(bank_config["prior_count"]))
                current_context = contexts[event["post_query_writes"][0]]
                feature = [
                    float(top_prediction[0]),
                    second,
                    float(top_prediction[0] - second),
                    float(prediction[top].mean()),
                    float(prediction[top].std()),
                    float(appearance[choice]),
                    float(appearance[top].max()),
                    float(np.where(top == choice)[0][0] / max(len(top) - 1, 1)),
                    len(bank) / capacity_value,
                    float(candidates[choice]["current_objective_improvement"]),
                    float(selected_pred_mean),
                    float(np.log1p(selected["pred_count"])),
                    float(selected_utility_mean),
                    float(np.log1p(selected["utility_count"])),
                    float(np.log1p(selected["frequency"])),
                    float((serial - selected["serial"]) / max(serial, 1)),
                    float(_score(selected, bank_config["policy"], bank_config)),
                    float(selected["scene"] == current_context["scene"]),
                ]
                target = float(candidates[choice]["future_utility"])
                feature_rows.append({
                    "fold": stream["fold"],
                    "order": order_name,
                    "episode": event["episode"],
                    "group": group_by_episode[event["episode"]],
                    "selected_context": selected["context_id"],
                    "future_utility": target,
                    "features": feature,
                })
                _update_history(
                    bank, event["episode"], lookup, bank_config["policy"], top_k,
                    selected["context_id"],
                )
                future_after = [
                    stream["events"][index]["episode"] for index in sequence[position + 1:]
                ]
                for context_id in event["post_query_writes"]:
                    serial += 1
                    _write(
                        bank, contexts[context_id], bank_config["policy"], capacity_value,
                        serial, bank_config, future_after, lookup, counts,
                    )

    matrix = np.asarray([row["features"] for row in feature_rows], dtype=np.float64)
    target = np.asarray([row["future_utility"] for row in feature_rows], dtype=np.float64)
    groups = np.asarray([row["group"] for row in feature_rows])
    feature_sets = {
        "full_bank_history": np.arange(17),
        "full_plus_scene_match": np.arange(18),
        "score_distribution_only": np.arange(9),
        "history_only": np.arange(8, 17),
        "raw_top_prediction": np.asarray([0]),
    }
    predictions = {name: np.empty(len(feature_rows)) for name in feature_sets}
    fold_rows = []
    for group in sorted(set(groups.tolist())):
        train, test = groups != group, groups == group
        diagnostics = {}
        for name, indices in feature_sets.items():
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=float(config["gate"]["ridge_alpha"])),
            )
            model.fit(matrix[train][:, indices], target[train])
            predictions[name][test] = model.predict(matrix[test][:, indices])
            diagnostics[name] = float(np.abs(predictions[name][test] - target[test]).mean())
        fold_rows.append({
            "held_out_component": group,
            "train_events": int(train.sum()),
            "test_events": int(test.sum()),
            "mae": diagnostics,
        })

    epsilon = float(table["utility_deadband"])
    primary_threshold = float(config["gate"]["primary_threshold"])
    metrics = {}
    for name, score in predictions.items():
        metrics[f"{name}_threshold_epsilon"] = _gate_metrics(
            feature_rows, score, primary_threshold, epsilon, group_by_episode,
        )
        metrics[f"{name}_threshold_zero"] = _gate_metrics(
            feature_rows, score, float(config["gate"]["control_threshold"]),
            epsilon, group_by_episode,
        )
    metrics.update({
        "uncalibrated_router": _gate_metrics(
            feature_rows, matrix[:, 0], 0.0, epsilon, group_by_episode,
        ),
        "current_objective_gate": _gate_metrics(
            feature_rows, matrix[:, 9], 0.0, epsilon, group_by_episode,
        ),
        "oracle_selected_or_reject": _gate_metrics(
            feature_rows, target, 0.0, epsilon, group_by_episode,
        ),
    })
    primary_name = "full_bank_history_threshold_epsilon"
    primary = metrics[primary_name]
    scene_variant = next(
        row for row in capacity["variants"]
        if row["policy"] == "scene_latest" and row["capacity"] == capacity_value
    )
    scene = scene_variant["summary"]["metrics"]["router_topk"]
    scene_rows = [
        row for row in capacity["rows"]
        if row["policy"] == "scene_latest" and row["capacity"] == capacity_value
    ]
    primary_component = {}
    scene_component = {}
    for group in sorted(set(groups.tolist())):
        primary_indices = np.flatnonzero(groups == group)
        accepted = predictions["full_bank_history"][primary_indices] > primary_threshold
        primary_component[group] = float(np.where(accepted, target[primary_indices], 0.0).mean())
        scene_component[group] = float(np.mean([
            row["router_topk"] for row in scene_rows if group_by_episode[row["episode"]] == group
        ]))
    difference = _bootstrap_difference(
        primary_component, scene_component, sorted(set(groups.tolist())),
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
        "stage": "stage2_bank_aware_acceptance_logo",
        "split": "train",
        "protocol_revision": "v1.4",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "current_target_and_past_delayed_history_only",
        "same_event_future_in_features": False,
        "grouping": "leave_one_physical_overlap_component_out",
        "components": len(set(groups.tolist())),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "feature_contract": {
            "dimensions": 18,
            "primary_excludes_scene_match": True,
            "query_or_future_input": False,
        },
        "label_health": {
            "beneficial": int(np.sum(target > epsilon)),
            "neutral": int(np.sum(np.abs(target) <= epsilon)),
            "harmful": int(np.sum(target < -epsilon)),
        },
        "candidate_utility_spearman": {
            name: float(spearmanr(score, target).statistic) for name, score in predictions.items()
        },
        "metrics": metrics,
        "scene_latest_capacity8": scene,
        "primary": primary_name,
        "primary_minus_scene_component_bootstrap": difference,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "folds": fold_rows,
        "rows": [{
            **row,
            **{name: float(score[index]) for name, score in predictions.items()},
        } for index, row in enumerate(feature_rows)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "label_health": result["label_health"],
        "spearman": result["candidate_utility_spearman"],
        "primary": primary, "scene": scene, "difference": difference,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
