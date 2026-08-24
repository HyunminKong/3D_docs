#!/usr/bin/env python3
"""Audit whether grouped OOF risk supervision is statistically identifiable."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _counts(rows: list[dict], epsilon: float) -> dict[str, int]:
    labels = [
        "beneficial" if row["future_utility"] > epsilon else
        "harmful" if row["future_utility"] < -epsilon else "neutral"
        for row in rows
    ]
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in ("beneficial", "neutral", "harmful")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", default="revisit3d/results/EXP-006/stage1_router_features_crossfit_train_v26.json",
    )
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage2_router_label_health_crossfit_train_v26.json",
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.features).read_text())
    if not (
        payload.get("split") == "train"
        and payload.get("validation_accessed") is False
        and payload.get("router_feature_contract", {}).get("query_or_future_input") is False
    ):
        raise RuntimeError("label audit requires leakage-safe train-only OOF features")
    rows = payload["router_features"]
    epsilon = float(payload["utility_epsilon"])
    folds = sorted({int(row["fold"]) for row in rows})
    fold_rows = []
    for fold in folds:
        train = [row for row in rows if int(row["fold"]) != fold]
        held_out = [row for row in rows if int(row["fold"]) == fold]
        train_counts, held_out_counts = _counts(train, epsilon), _counts(held_out, epsilon)
        fold_rows.append({
            "fold": fold,
            "train_candidates": len(train),
            "held_out_candidates": len(held_out),
            "train_label_counts": train_counts,
            "held_out_label_counts": held_out_counts,
            "risk_train_has_beneficial_and_harmful": (
                train_counts["beneficial"] > 0 and train_counts["harmful"] > 0
            ),
        })
    harmful_rows = [row for row in rows if row["future_utility"] < -epsilon]
    harmful_folds = sorted({int(row["fold"]) for row in harmful_rows})
    harmful_episodes = sorted({row["episode"] for row in harmful_rows})
    result = {
        "experiment": "EXP-006",
        "stage": "stage2_router_label_health",
        "split": "train",
        "protocol_revision": payload["protocol_revision"],
        "source_features": args.features,
        "validation_accessed": False,
        "query_or_future_router_input": False,
        "utility_epsilon": epsilon,
        "overall_label_counts": _counts(rows, epsilon),
        "harmful_overlap_fold_count": len(harmful_folds),
        "harmful_folds": harmful_folds,
        "harmful_episode_count": len(harmful_episodes),
        "harmful_episodes": harmful_episodes,
        "folds": fold_rows,
        "risk_identifiability_gate": {
            "definition": "every grouped OOF training partition contains both beneficial and harmful labels, and harmful labels occur in at least two held-out overlap folds",
            "passed": (
                all(row["risk_train_has_beneficial_and_harmful"] for row in fold_rows)
                and len(harmful_folds) >= 2
            ),
        },
        "interpretation": (
            "A utility regressor can be probed, but grouped out-of-fold risk classification is not "
            "identifiable if one held-out physical-overlap group contains all harmful examples."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "out": str(output),
        "overall": result["overall_label_counts"],
        "harmful_folds": harmful_folds,
        "passed": result["risk_identifiability_gate"]["passed"],
    }))


if __name__ == "__main__":
    main()
