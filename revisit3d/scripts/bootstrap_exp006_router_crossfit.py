#!/usr/bin/env python3
"""Overlap-component bootstrap for EXP-006 out-of-fold router feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from revisit3d.experiments import grouped_folds


def _interval(values: np.ndarray) -> dict:
    low, high = np.percentile(values, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--router", default="revisit3d/results/EXP-006/stage2_router_similarity_controls_crossfit_train_v26.json",
    )
    parser.add_argument(
        "--features", default="revisit3d/results/EXP-006/stage1_router_features_crossfit_train_v26.json",
    )
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage2_router_bootstrap_crossfit_train_v26.json",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=600006)
    args = parser.parse_args()
    router = json.loads(Path(args.router).read_text())
    features = json.loads(Path(args.features).read_text())
    if not (
        router.get("split") == features.get("split") == "train"
        and router.get("validation_accessed") is False
        and features.get("validation_accessed") is False
    ):
        raise RuntimeError("router bootstrap requires train-only out-of-fold inputs")
    records = [
        row for row in json.loads(Path(args.manifest).read_text()) if row["split"] == "train"
    ]
    _, group_of = grouped_folds(records, 5, 600)
    group_by_episode = {record["episode_id"]: group_of[index] for index, record in enumerate(records)}
    router_rows = router["rows"]
    feature_rows = features["router_features"]
    mean_pool = {
        row["episode"]: row["utility"]
        for row in features["rows"] if row["condition"] == "visual_mean_pool"
    }
    model_names = (
        "ridge_full", "ridge_full_without_geometry", "ridge_descriptor_only",
        "ridge_online_scalars", "ridge_geometry_only",
    )
    episode_values = {}
    for episode in sorted({row["episode"] for row in router_rows}):
        candidates = [row for row in router_rows if row["episode"] == episode]
        utilities = np.asarray([row["future_utility"] for row in candidates])
        values = {
            name: float(utilities[np.argmax([row[name] for row in candidates])])
            for name in model_names
        }
        source_candidates = [row for row in feature_rows if row["episode"] == episode]
        values.update({
            "mean_pool": float(mean_pool[episode]),
            "random_expectation": float(np.mean([row["future_utility"] for row in source_candidates])),
            "matched_a": float(next(
                row["future_utility"] for row in source_candidates if row["candidate"] == "matched_a"
            )),
            "oracle": max(0.0, float(utilities.max())),
        })
        episode_values[episode] = values
    groups = sorted(set(group_by_episode.values()))
    episodes_by_group = {
        group: [episode for episode in episode_values if group_by_episode[episode] == group]
        for group in groups
    }
    rng = np.random.default_rng(args.seed)
    methods = tuple(next(iter(episode_values.values())).keys())
    draws = {method: [] for method in methods}
    differences = {
        "ridge_full_minus_mean_pool": [],
        "ridge_full_minus_random_expectation": [],
        "ridge_full_minus_matched_a": [],
        "ridge_online_minus_mean_pool": [],
        "ridge_descriptor_minus_mean_pool": [],
        "ridge_descriptor_minus_random_expectation": [],
        "ridge_descriptor_minus_matched_a": [],
        "ridge_no_geometry_minus_mean_pool": [],
        "ridge_no_geometry_minus_random_expectation": [],
        "ridge_no_geometry_minus_matched_a": [],
    }
    for _ in range(args.samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_episodes = [
            episode for group in sampled_groups for episode in episodes_by_group[str(group)]
        ]
        statistic = {
            method: float(np.mean([episode_values[episode][method] for episode in sampled_episodes]))
            for method in methods
        }
        for method, value in statistic.items():
            draws[method].append(value)
        differences["ridge_full_minus_mean_pool"].append(
            statistic["ridge_full"] - statistic["mean_pool"]
        )
        differences["ridge_full_minus_random_expectation"].append(
            statistic["ridge_full"] - statistic["random_expectation"]
        )
        differences["ridge_full_minus_matched_a"].append(
            statistic["ridge_full"] - statistic["matched_a"]
        )
        differences["ridge_online_minus_mean_pool"].append(
            statistic["ridge_online_scalars"] - statistic["mean_pool"]
        )
        differences["ridge_descriptor_minus_mean_pool"].append(
            statistic["ridge_descriptor_only"] - statistic["mean_pool"]
        )
        differences["ridge_descriptor_minus_random_expectation"].append(
            statistic["ridge_descriptor_only"] - statistic["random_expectation"]
        )
        differences["ridge_descriptor_minus_matched_a"].append(
            statistic["ridge_descriptor_only"] - statistic["matched_a"]
        )
        differences["ridge_no_geometry_minus_mean_pool"].append(
            statistic["ridge_full_without_geometry"] - statistic["mean_pool"]
        )
        differences["ridge_no_geometry_minus_random_expectation"].append(
            statistic["ridge_full_without_geometry"] - statistic["random_expectation"]
        )
        differences["ridge_no_geometry_minus_matched_a"].append(
            statistic["ridge_full_without_geometry"] - statistic["matched_a"]
        )
    result = {
        "experiment": "EXP-006", "stage": "stage2_router_grouped_bootstrap",
        "split": "train", "protocol_revision": router["protocol_revision"],
        "source_router": args.router, "source_features": args.features,
        "validation_accessed": False, "group_unit": "physical_overlap_component",
        "groups": len(groups), "episodes": len(episode_values),
        "bootstrap_samples": args.samples, "bootstrap_seed": args.seed,
        "method_intervals": {
            method: _interval(np.asarray(values)) for method, values in draws.items()
        },
        "difference_intervals": {
            name: _interval(np.asarray(values)) for name, values in differences.items()
        },
        "episode_values": episode_values,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "out": str(output), "method_intervals": result["method_intervals"],
        "difference_intervals": result["difference_intervals"],
    }), flush=True)


if __name__ == "__main__":
    main()
