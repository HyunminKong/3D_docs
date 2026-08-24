#!/usr/bin/env python3
"""Evaluate the selected atom bank with unique writes in true capture-time order."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.simulate_exp007_causal_bank import _add, _evaluate, _sha256
from revisit3d.scripts.simulate_exp007_token_bucket import _write_token
from revisit3d.scripts.simulate_exp007_utility_consolidation import _summary, _update_history


TIMESTAMP = re.compile(r"__(\d+)\.jpg$")


def _frame_times(scene_root: Path, scene: str) -> list[int]:
    payload = json.loads((scene_root / scene / "opencv_cameras.json").read_text())
    values = []
    for frame in payload["frames"]:
        match = TIMESTAMP.search(frame["file_path"])
        if match is None:
            raise RuntimeError(f"timestamp missing from {frame['file_path']}")
        values.append(int(match.group(1)))
    if any(left >= right for left, right in zip(values, values[1:])):
        raise RuntimeError(f"non-monotonic camera timestamps in {scene}")
    return values


def _empty_evaluation() -> dict:
    return {
        "router_topk": 0.0, "router_all": 0.0, "appearance_top1": 0.0,
        "random_expectation": 0.0, "oracle_topk": 0.0, "oracle_all": 0.0,
        "router_accepted": False, "router_all_accepted": False,
        "selected_context": None, "oracle_context": None, "oracle_in_topk": False,
        "bank_records": 0, "bank_unique_contexts": 0,
        "appearance_comparisons": 0, "router_comparisons": 0,
    }


def _assert_duplicate_targets_identical(stream: dict, target_events: dict[str, list[dict]]) -> None:
    lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
    context_ids = [row["context_id"] for row in stream["contexts"]]
    for events in target_events.values():
        if len(events) < 2:
            continue
        reference = events[0]["episode"]
        for event in events[1:]:
            for context_id in context_ids:
                left = lookup[(reference, context_id)]
                right = lookup[(event["episode"], context_id)]
                if not (
                    left["future_utility"] == right["future_utility"]
                    and left["predicted_utility"] == right["predicted_utility"]
                ):
                    raise RuntimeError("duplicate target contexts have inconsistent utility rows")


def _bootstrap(
    left: dict[str, float], right: dict[str, float], groups: list[str], samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean([left[group] - right[group] for group in rng.choice(groups, len(groups), replace=True)])
        for _ in range(samples)
    ])
    return {
        "bootstrap_mean": float(draws.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [.025, .975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-008_true_timestamp_stream_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-008 result already exists: {output}")

    table_path = Path(config["source"]["oof_utility_table"])
    bucket_path = Path(config["source"]["crossfit_frozen_bucket"])
    table = json.loads(table_path.read_text())
    bucket = json.loads(bucket_path.read_text())
    if not (
        table.get("split") == bucket.get("split") == "train"
        and table.get("validation_accessed") is False
        and bucket.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
        and bucket.get("bucket_key_source") == "frozen_foundation_crossfit_pca"
    ):
        raise RuntimeError("EXP-008 requires locked train-only fold-local sources")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    record_by_episode = {row["episode_id"]: row for row in records}
    _, component = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {row["episode_id"]: component[index] for index, row in enumerate(records)}
    scene_root = Path(config["data"]["scene_root"])
    time_cache: dict[str, list[int]] = {}

    probability_by_fold: dict[int, dict[tuple[str, str], float]] = {}
    for row in bucket["pair_rows"]:
        fold = int(row["fold"])
        probability_by_fold.setdefault(fold, {})[
            tuple(sorted((row["left"], row["right"])))
        ] = float(row["oof_probability"])

    prepared = []
    representative_episodes = []
    for stream in table["streams"]:
        fold = int(stream["fold"])
        contexts = {row["context_id"]: dict(row) for row in stream["contexts"]}
        for context in contexts.values():
            scene = context["scene"]
            if scene not in time_cache:
                time_cache[scene] = _frame_times(scene_root, scene)
            indices = [int(index) for index in context["frames"]]
            context["start_timestamp_us"] = time_cache[scene][min(indices)]
            context["end_timestamp_us"] = time_cache[scene][max(indices)]

        target_events: dict[str, list[dict]] = {}
        for event in stream["events"]:
            target_events.setdefault(event["post_query_writes"][0], []).append(event)
        _assert_duplicate_targets_identical(stream, target_events)
        representatives = {
            context_id: sorted(events, key=lambda row: row["episode"])[0]
            for context_id, events in target_events.items()
        }
        representative_episodes.extend(row["episode"] for row in representatives.values())
        timeline = sorted(
            contexts.values(), key=lambda row: (row["end_timestamp_us"], row["context_id"]),
        )
        prepared.append({
            "fold": fold, "contexts": contexts, "timeline": timeline,
            "representatives": representatives,
            "lookup": {(row["episode"], row["context_id"]): row for row in stream["pairs"]},
        })

    variants = []
    rows_by_policy: dict[str, list[dict]] = {}
    oracle_reference = {}
    policies = config["bank"]["policies"]
    for policy_index, policy in enumerate(policies):
        rows = []
        aggregate = {
            "writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0,
            "true_bucket_merges": 0, "false_bucket_merges": 0,
        }
        for stream in prepared:
            fold = stream["fold"]
            bank = []
            seen_writes = [0]
            rng = random.Random(int(config["seed"]) + 1009 * policy_index + fold)
            serial = int(config["seed"]) + 100000 * fold
            for position, context in enumerate(stream["timeline"]):
                event = stream["representatives"].get(context["context_id"])
                if event is not None:
                    evaluation = (
                        _evaluate(
                            bank, event["episode"], stream["lookup"],
                            int(config["bank"]["top_k"]),
                            float(config["bank"]["utility_threshold"]),
                        ) if bank else _empty_evaluation()
                    )
                    key = (fold, event["episode"])
                    if policy == "unbounded_unique_context":
                        oracle_reference[key] = evaluation["oracle_all"]
                    else:
                        evaluation["causal_unbounded_oracle"] = oracle_reference[key]
                    rows.append({
                        "fold": fold, "policy": policy, "capacity": (
                            None if policy == "unbounded_unique_context" else int(config["bank"]["capacity"])
                        ),
                        "order": "true_timestamp", "stream_position": position,
                        "timestamp_us": context["end_timestamp_us"],
                        "episode": event["episode"], **evaluation,
                    })
                    if policy == "frozen_bucket_predicted_history" and bank:
                        _update_history(
                            bank, event["episode"], stream["lookup"], "predicted_history",
                            int(config["bank"]["top_k"]), evaluation["selected_context"],
                        )

                serial += 1
                if policy == "frozen_bucket_predicted_history":
                    _write_token(
                        bank, context, policy, int(config["bank"]["capacity"]), serial,
                        probability_by_fold[fold],
                        float(config["bank"]["bucket_probability_threshold"]),
                        config["bank"], aggregate,
                    )
                else:
                    base_policy = policy
                    _add(
                        bank, context, policy=base_policy,
                        capacity=None if policy == "unbounded_unique_context" else int(config["bank"]["capacity"]),
                        rng=rng, counters=aggregate, seen_writes=seen_writes,
                    )

        if policy == "unbounded_unique_context":
            for row in rows:
                row["causal_unbounded_oracle"] = row["oracle_all"]
        metric = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variants.append({"policy": policy, "summary": metric, "counts": aggregate})
        rows_by_policy[policy] = rows
        print(json.dumps({"policy": policy, "summary": metric, "counts": aggregate}), flush=True)

    primary_name = config["success"]["primary_policy"]
    comparator_name = config["success"]["comparator_policy"]
    oracle_name = config["success"]["oracle_bucket_policy"]
    by_name = {row["policy"]: row for row in variants}
    primary = by_name[primary_name]["summary"]
    comparator = by_name[comparator_name]["summary"]
    oracle = by_name[oracle_name]["summary"]
    groups = sorted({group_by_episode[episode] for episode in representative_episodes})

    def component_means(policy: str) -> dict[str, float]:
        return {
            group: float(np.mean([
                row["router_topk"] for row in rows_by_policy[policy]
                if group_by_episode[row["episode"]] == group
            ]))
            for group in groups
        }

    difference = _bootstrap(
        component_means(primary_name), component_means(comparator_name), groups,
        int(config["statistics"]["bootstrap_samples"]),
        int(config["statistics"]["bootstrap_seed"]),
    )
    retention = (
        primary["mean_utility"] / oracle["mean_utility"]
        if oracle["mean_utility"] > 0 else float("nan")
    )
    passed = (
        primary["mean_utility"] > comparator["mean_utility"]
        and primary["harmful_rate"] <= comparator["harmful_rate"]
        and retention >= float(config["success"]["minimum_oracle_bucket_utility_retention"])
    )
    result = {
        "experiment": "EXP-008", "stage": "stage0_true_timestamp_stream",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False,
        "future_utility_role": "offline_evaluation_only",
        "true_capture_timestamp_order": True,
        "unique_context_write_once": True,
        "fold_local_memory_coordinates": True,
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "source_table": str(table_path), "source_bucket": str(bucket_path),
        "unique_target_events": len(representative_episodes),
        "variants": variants,
        "primary_minus_comparator_component_bootstrap": difference,
        "primary_oracle_bucket_utility_retention": retention,
        "registered_gate": {"passed": bool(passed)},
        "rows": [row for policy_rows in rows_by_policy.values() for row in policy_rows],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "events": len(representative_episodes),
        "retention": retention, "difference": difference,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
