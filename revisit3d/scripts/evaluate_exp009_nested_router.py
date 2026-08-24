#!/usr/bin/env python3
"""Nested component-crossfit utility routing for EXP-009 unseen train candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import PRIMARY_SCALAR_INDICES, primary_feature_columns
from revisit3d.scripts.evaluate_exp006_validation import _bootstrap, _summarize


def _model(config: dict):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=int(config["router"]["pca_components"]), random_state=int(config["seed"])),
        Ridge(alpha=float(config["router"]["ridge_alpha"])),
    )


def _episode_winners(rows: list[dict], prediction: np.ndarray, indices: np.ndarray) -> list[dict]:
    by_episode = {}
    for index in indices:
        by_episode.setdefault(rows[int(index)]["episode"], []).append(int(index))
    output = []
    for episode, candidates in by_episode.items():
        choice = max(candidates, key=lambda index: float(prediction[index]))
        output.append({
            "episode": episode, "score": float(prediction[choice]),
            "utility": float(rows[choice]["future_utility"]),
            "candidate": rows[choice]["candidate"],
        })
    return output


def _calibrate_threshold(
    winners: list[dict], visual_by_episode: dict[str, float], epsilon: float, minimum_acceptance: float,
) -> dict:
    visual_harm = float(np.mean([
        visual_by_episode[row["episode"]] < -epsilon for row in winners
    ]))
    scores = np.asarray([row["score"] for row in winners])
    thresholds = np.unique(np.concatenate(([
        float(scores.min() - 1.0), 0.0,
    ], scores, [float(scores.max() + 1.0)])))
    candidates = []
    for threshold in thresholds:
        utility = np.asarray([
            row["utility"] if row["score"] > threshold else 0.0 for row in winners
        ])
        acceptance = float(np.mean(scores > threshold))
        harm = float(np.mean(utility < -epsilon))
        if acceptance >= minimum_acceptance and harm <= visual_harm:
            candidates.append({
                "threshold": float(threshold), "mean_utility": float(utility.mean()),
                "harmful_rate": harm, "acceptance": acceptance,
            })
    if not candidates:
        return {
            "threshold": float(scores.max() + 1.0), "mean_utility": 0.0,
            "harmful_rate": 0.0, "acceptance": 0.0,
            "fallback_reject_all": True, "visual_harm_constraint": visual_harm,
        }
    best = max(candidates, key=lambda row: (row["mean_utility"], -row["harmful_rate"], row["threshold"]))
    return {**best, "fallback_reject_all": False, "visual_harm_constraint": visual_harm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_nested_router_v16.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-009 nested-router result already exists: {output}")
    candidate = json.loads(Path(config["source"]["candidate_cache"]).read_text())
    transfer = json.loads(Path(config["source"]["locked_transfer_result"]).read_text())
    if not (
        candidate.get("split") == transfer.get("split") == "train"
        and candidate.get("validation_accessed") is False
        and transfer.get("validation_accessed") is False
        and candidate.get("test_accessed") is False
        and transfer.get("test_accessed") is False
        and candidate.get("query_or_future_router_input") is False
        and transfer.get("query_or_future_router_input") is False
    ):
        raise RuntimeError("nested router requires locked train-only observable candidates")
    rows = candidate["rows"]
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["component"] for row in rows])
    components = sorted(set(groups.tolist()))
    episodes = sorted({row["episode"] for row in rows})
    group_by_episode = {
        row["episode"]: row["component"] for row in rows
    }
    if len(rows) != 5 * len(episodes) or len(components) != transfer["components"]:
        raise RuntimeError("candidate pool or component contract changed")
    columns = primary_feature_columns(
        int(config["router"]["descriptor_dimensions"]),
        tuple(config["router"]["primary_scalar_indices"]),
    )
    if tuple(config["router"]["primary_scalar_indices"]) != PRIMARY_SCALAR_INDICES:
        raise RuntimeError("primary scalar feature contract changed")
    features = matrix[:, columns]
    visual_by_episode = {
        row["episode"]: float(row["visual_mean_utility"])
        for row in transfer["selection_rows"]
    }
    old_router_by_episode = {
        row["episode"]: float(row["router_utility"])
        for row in transfer["selection_rows"]
    }
    epsilon = float(config["router"]["utility_deadband"])
    minimum_acceptance = float(config["router"]["minimum_acceptance"])

    oof_prediction = np.empty(len(rows), dtype=np.float64)
    threshold_by_component = {}
    primary_values = {}
    primary_accepted = {}
    ungated_values = {}
    ungated_accepted = {}
    selections = []
    for held_out in components:
        outer_train = groups != held_out
        outer_test = groups == held_out
        inner_prediction = np.full(len(rows), np.nan, dtype=np.float64)
        for inner_held_out in [group for group in components if group != held_out]:
            inner_train = (groups != held_out) & (groups != inner_held_out)
            inner_test = groups == inner_held_out
            model = _model(config)
            model.fit(features[inner_train], utility[inner_train])
            inner_prediction[inner_test] = model.predict(features[inner_test])
        inner_indices = np.flatnonzero(outer_train)
        if not np.isfinite(inner_prediction[inner_indices]).all():
            raise RuntimeError("nested calibration prediction is incomplete")
        calibration = _calibrate_threshold(
            _episode_winners(rows, inner_prediction, inner_indices),
            visual_by_episode, epsilon, minimum_acceptance,
        )
        threshold_by_component[held_out] = calibration
        outer_model = _model(config)
        outer_model.fit(features[outer_train], utility[outer_train])
        oof_prediction[outer_test] = outer_model.predict(features[outer_test])
        for winner in _episode_winners(rows, oof_prediction, np.flatnonzero(outer_test)):
            take = winner["score"] > calibration["threshold"]
            primary_values[winner["episode"]] = winner["utility"] if take else 0.0
            primary_accepted[winner["episode"]] = take
            ungated_values[winner["episode"]] = winner["utility"]
            ungated_accepted[winner["episode"]] = True
            selections.append({
                **winner, "component": held_out, "threshold": calibration["threshold"],
                "accepted": take,
            })

    if set(primary_values) != set(episodes) or not np.isfinite(oof_prediction).all():
        raise RuntimeError("outer OOF routing did not cover every episode/candidate")
    methods = {
        "router": primary_values,
        "ungated_new_router": ungated_values,
        "old_locked_router": old_router_by_episode,
        "visual_mean": visual_by_episode,
    }
    accepted = {
        "router": primary_accepted,
        "ungated_new_router": ungated_accepted,
        "old_locked_router": {episode: True for episode in episodes},
        "visual_mean": {episode: True for episode in episodes},
    }
    metrics = {
        name: _summarize(values, accepted[name], group_by_episode, epsilon)
        for name, values in methods.items()
    }
    bootstrap = _bootstrap(
        methods, group_by_episode,
        int(config["statistics"]["bootstrap_samples"]), int(config["statistics"]["bootstrap_seed"]),
    )
    difference = bootstrap["differences"]["router_minus_visual_mean"]
    checks = {
        "beats_visual_mean": metrics["router"]["mean_selected_utility"]
        > metrics["visual_mean"]["mean_selected_utility"],
        "harm_not_above_visual": metrics["router"]["harmful_rate"]
        <= metrics["visual_mean"]["harmful_rate"],
        "nontrivial_acceptance": metrics["router"]["accept_rate"]
        >= float(config["success"]["minimum_acceptance"]),
        "positive_component_interval": difference["ci95"][0] > 0.0,
    }
    result = {
        "experiment": "EXP-009", "stage": "stage6_nested_component_router",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False,
        "config": str(config_path), "episodes": len(episodes),
        "candidates": len(rows), "components": len(components),
        "feature_columns": columns, "threshold_by_component": threshold_by_component,
        "metrics": metrics, "bootstrap": bootstrap, "selections": selections,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "metrics": metrics,
        "difference": difference, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
