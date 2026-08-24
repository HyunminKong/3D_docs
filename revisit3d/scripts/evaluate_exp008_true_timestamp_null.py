#!/usr/bin/env python3
"""Matched probability-permutation null for EXP-008 true timestamp replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.evaluate_exp008_true_timestamp_stream import (
    _assert_duplicate_targets_identical,
    _empty_evaluation,
    _frame_times,
)
from revisit3d.scripts.simulate_exp007_causal_bank import _evaluate, _sha256
from revisit3d.scripts.simulate_exp007_token_bucket import _write_token
from revisit3d.scripts.simulate_exp007_utility_consolidation import _summary, _update_history


def _prepare(table: dict, scene_root: Path) -> list[dict]:
    time_cache: dict[str, list[int]] = {}
    prepared = []
    for stream in table["streams"]:
        contexts = {row["context_id"]: dict(row) for row in stream["contexts"]}
        for context in contexts.values():
            scene = context["scene"]
            if scene not in time_cache:
                time_cache[scene] = _frame_times(scene_root, scene)
            indices = [int(index) for index in context["frames"]]
            context["end_timestamp_us"] = time_cache[scene][max(indices)]
        target_events: dict[str, list[dict]] = {}
        for event in stream["events"]:
            target_events.setdefault(event["post_query_writes"][0], []).append(event)
        _assert_duplicate_targets_identical(stream, target_events)
        prepared.append({
            "fold": int(stream["fold"]),
            "timeline": sorted(
                contexts.values(), key=lambda row: (row["end_timestamp_us"], row["context_id"]),
            ),
            "representatives": {
                context_id: sorted(events, key=lambda row: row["episode"])[0]
                for context_id, events in target_events.items()
            },
            "lookup": {(row["episode"], row["context_id"]): row for row in stream["pairs"]},
        })
    return prepared


def _probability_map(pair_rows: list[dict], values: np.ndarray) -> dict:
    result = {}
    for row, value in zip(pair_rows, values):
        fold = int(row["fold"])
        result.setdefault(fold, {})[
            tuple(sorted((row["left"], row["right"])))
        ] = float(value)
    return result


def _simulate(
    prepared: list[dict], probability: dict, config: dict,
    group_by_episode: dict[str, str], oracle_reference: dict,
) -> dict:
    rows = []
    for stream in prepared:
        fold = stream["fold"]
        bank = []
        counts = {
            "writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0,
            "true_bucket_merges": 0, "false_bucket_merges": 0,
        }
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
                evaluation["causal_unbounded_oracle"] = oracle_reference[(fold, event["episode"])]
                rows.append({
                    "fold": fold, "order": "true_timestamp", "episode": event["episode"],
                    "stream_position": position, **evaluation,
                })
                if bank:
                    _update_history(
                        bank, event["episode"], stream["lookup"], "predicted_history",
                        int(config["bank"]["top_k"]), evaluation["selected_context"],
                    )
            serial += 1
            _write_token(
                bank, context, "token_bucket_predicted_history",
                int(config["bank"]["capacity"]), serial, probability[fold],
                float(config["bank"]["bucket_probability_threshold"]),
                config["bank"], counts,
            )
    return _summary(rows, float(config["utility_deadband"]), group_by_episode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-008_true_timestamp_null_v11.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-008 null result already exists: {output}")

    table = json.loads(Path(config["source"]["oof_utility_table"]).read_text())
    bucket = json.loads(Path(config["source"]["crossfit_frozen_bucket"]).read_text())
    true_time_path = Path(config["source"]["true_timestamp_result"])
    true_time = json.loads(true_time_path.read_text())
    if not (
        table.get("validation_accessed") is False
        and bucket.get("validation_accessed") is False
        and true_time.get("validation_accessed") is False
        and true_time.get("protocol_revision") == "v1.0"
        and true_time.get("true_capture_timestamp_order") is True
    ):
        raise RuntimeError("EXP-008 v1.1 requires the locked train-only Stage-0 result")
    config["utility_deadband"] = table["utility_deadband"]

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, components = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {row["episode_id"]: components[index] for index, row in enumerate(records)}
    prepared = _prepare(table, Path(config["data"]["scene_root"]))
    oracle_reference = {
        (int(row["fold"]), row["episode"]): row["causal_unbounded_oracle"]
        for row in true_time["rows"] if row["policy"] == "unbounded_unique_context"
    }
    observed = next(
        row["summary"] for row in true_time["variants"]
        if row["policy"] == config["bank"]["policy"]
    )
    appearance = next(
        row["summary"] for row in true_time["variants"]
        if row["policy"] == "appearance_diversity"
    )

    pair_rows = bucket["pair_rows"]
    original = np.asarray([row["oof_probability"] for row in pair_rows], dtype=np.float64)
    pair_fold = np.asarray([int(row["fold"]) for row in pair_rows])
    rng = np.random.default_rng(int(config["seed"]))
    metrics = []
    for index in range(int(config["permutation_null"]["samples"])):
        shuffled = original.copy()
        for fold in sorted(set(pair_fold.tolist())):
            mask = np.flatnonzero(pair_fold == fold)
            shuffled[mask] = rng.permutation(shuffled[mask])
        metrics.append(_simulate(
            prepared, _probability_map(pair_rows, shuffled), config,
            group_by_episode, oracle_reference,
        ))
        if (index + 1) % 100 == 0:
            print(json.dumps({"completed": index + 1}), flush=True)

    utility = np.asarray([row["mean_utility"] for row in metrics])
    harm = np.asarray([row["harmful_rate"] for row in metrics])
    p_value = float((1 + (utility >= observed["mean_utility"]).sum()) / (len(utility) + 1))
    passed = (
        p_value < float(config["success"]["one_sided_alpha"])
        and observed["harmful_rate"] <= appearance["harmful_rate"]
    )
    result = {
        "experiment": "EXP-008", "stage": "stage1_true_timestamp_bucket_null",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "validation_accessed": False, "test_accessed": False,
        "true_capture_timestamp_order": True,
        "config": str(config_path), "config_sha256": _sha256(config_path),
        "source_true_timestamp_result": str(true_time_path),
        "null_contract": "shuffle OOF pair probabilities within fold; preserve score distribution",
        "observed": observed, "appearance_diversity": appearance,
        "null": {
            "samples": len(metrics),
            "mean_utility_mean": float(utility.mean()),
            "mean_utility_q025_q50_q975": [float(value) for value in np.quantile(utility, [.025, .5, .975])],
            "harmful_rate_mean": float(harm.mean()),
            "harmful_rate_q025_q50_q975": [float(value) for value in np.quantile(harm, [.025, .5, .975])],
        },
        "one_sided_utility_p": p_value,
        "observed_utility_percentile": float((utility < observed["mean_utility"]).mean()),
        "registered_gate": {"passed": bool(passed)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
