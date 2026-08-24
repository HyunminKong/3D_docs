#!/usr/bin/env python3
"""Grouped out-of-fold feasibility test for observable visual-memory routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _selection_metrics(
    rows: list[dict], predictions: np.ndarray | None, epsilon: float, *, accept_threshold: float = 0.0,
) -> dict:
    selected, oracle, regret, accepted = [], [], [], []
    for episode in sorted({row["episode"] for row in rows}):
        subset = [row for row in rows if row["episode"] == episode]
        utility = np.asarray([row["future_utility"] for row in subset], dtype=np.float64)
        oracle_value = max(0.0, float(utility.max()))
        if predictions is None:
            score = np.asarray([
                row["current_objective_improvement"] for row in subset
            ], dtype=np.float64)
        else:
            score = np.asarray([predictions[row["row_index"]] for row in subset], dtype=np.float64)
        choice = int(score.argmax())
        accept = bool(score[choice] > accept_threshold)
        value = float(utility[choice]) if accept else 0.0
        selected.append(value)
        oracle.append(oracle_value)
        regret.append(oracle_value - value)
        accepted.append(accept)
    values = np.asarray(selected)
    return {
        "episodes": len(values),
        "mean_selected_utility": float(values.mean()),
        "median_selected_utility": float(np.median(values)),
        "beneficial_rate": float(np.mean(values > epsilon)),
        "harmful_rate": float(np.mean(values < -epsilon)),
        "accept_rate": float(np.mean(accepted)),
        "accept_threshold": accept_threshold,
        "mean_oracle_utility": float(np.mean(oracle)),
        "mean_future_utility_regret": float(np.mean(regret)),
    }


def _fixed_candidate(rows: list[dict], label: str, epsilon: float) -> dict:
    values = np.asarray([
        row["future_utility"] for row in rows if row["candidate"] == label
    ], dtype=np.float64)
    return {
        "episodes": len(values), "mean_selected_utility": float(values.mean()),
        "median_selected_utility": float(np.median(values)),
        "beneficial_rate": float(np.mean(values > epsilon)),
        "harmful_rate": float(np.mean(values < -epsilon)), "accept_rate": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", default="revisit3d/results/EXP-006/stage1_router_features_crossfit_train_v26.json",
    )
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage2_router_feasibility_crossfit_train_v26.json",
    )
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    payload = json.loads(Path(args.features).read_text())
    if not (
        payload.get("split") == "train"
        and payload.get("validation_accessed") is False
        and payload.get("router_feature_contract", {}).get("query_or_future_input") is False
    ):
        raise RuntimeError("router features violate the train-only observable-input contract")
    rows = payload["router_features"]
    feature_dim = int(payload["router_feature_contract"]["total_dimensions"])
    scalar_dim = int(payload["router_feature_contract"]["observable_scalar_dimensions"])
    if len(rows) % 5 or any(len(row["features"]) != feature_dim for row in rows):
        raise RuntimeError("expected five candidates per episode with the registered feature dimension")
    for index, row in enumerate(rows):
        row["row_index"] = index
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise RuntimeError("non-finite router feature or target")
    if scalar_dim == 24:
        nonalignment = np.concatenate((features[:, 256:268], features[:, 272:280]), axis=1)
        full_without_alignment = np.concatenate((features[:, :268], features[:, 272:280]), axis=1)
        alignment = features[:, 268:272]
    elif scalar_dim == 16:
        nonalignment = features[:, 256:268]
        full_without_alignment = features[:, :268]
        alignment = features[:, 268:272]
    else:
        raise RuntimeError(f"unsupported observable scalar contract: {scalar_dim}")
    feature_sets = {
        "ridge_full": features,
        "ridge_full_without_geometry": full_without_alignment,
        "ridge_descriptor_only": features[:, :256],
        "ridge_online_scalars": nonalignment,
        "ridge_online_without_geometry": nonalignment,
        "ridge_geometry_only": alignment,
    }
    predictions = {name: np.empty_like(targets) for name in feature_sets}
    fold_rows = []
    for fold in sorted(set(folds.tolist())):
        train, test = folds != fold, folds == fold
        fold_mae = {}
        for name, matrix in feature_sets.items():
            if matrix.shape[1] > 32:
                components = min(args.pca_components, int(train.sum()) - 1, matrix.shape[1])
                model = make_pipeline(
                    StandardScaler(), PCA(n_components=components, random_state=600),
                    Ridge(alpha=args.ridge_alpha),
                )
            else:
                model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
            model.fit(matrix[train], targets[train])
            predictions[name][test] = model.predict(matrix[test])
            fold_mae[name] = float(np.abs(predictions[name][test] - targets[test]).mean())
        fold_rows.append({
            "fold": fold, "train_candidates": int(train.sum()), "test_candidates": int(test.sum()),
            "target_mean": float(targets[test].mean()), "prediction_mae": fold_mae,
        })
    prediction_full = predictions["ridge_full"]
    prediction_online = predictions["ridge_online_scalars"]
    epsilon = float(payload["utility_epsilon"])
    mean_pool_values = np.asarray([
        row["utility"] for row in payload["rows"] if row["condition"] == "visual_mean_pool"
    ], dtype=np.float64)
    random_expected = []
    for episode in sorted({row["episode"] for row in rows}):
        random_expected.append(np.mean([
            row["future_utility"] for row in rows if row["episode"] == episode
        ]))
    metrics = {
        **{
            name: _selection_metrics(rows, prediction, epsilon)
            for name, prediction in predictions.items()
        },
        "ridge_consensus_mean": _selection_metrics(
            rows, 0.5 * (prediction_full + prediction_online), epsilon,
        ),
        "ridge_conservative_min": _selection_metrics(
            rows, np.minimum(prediction_full, prediction_online), epsilon,
        ),
        "ridge_online_scalars_deadband_accept": _selection_metrics(
            rows, prediction_online, epsilon, accept_threshold=epsilon,
        ),
        "current_objective_rerank": _selection_metrics(rows, None, epsilon),
        "appearance_similarity_retrieval": _selection_metrics(
            rows, features[:, 267], epsilon, accept_threshold=-1e6,
        ),
        "geometry_quality_retrieval": _selection_metrics(
            rows,
            np.where(features[:, 268] > 0.5, features[:, 269] - features[:, 270] / 2.5, -1e6),
            epsilon,
            accept_threshold=-1e6,
        ),
        "matched_a": _fixed_candidate(rows, "matched_a", epsilon),
        "random_candidate_expectation": {
            "episodes": len(random_expected),
            "mean_selected_utility": float(np.mean(random_expected)),
        },
        "visual_mean_pool": {
            "episodes": len(mean_pool_values),
            "mean_selected_utility": float(mean_pool_values.mean()),
            "median_selected_utility": float(np.median(mean_pool_values)),
            "beneficial_rate": float(np.mean(mean_pool_values > epsilon)),
            "harmful_rate": float(np.mean(mean_pool_values < -epsilon)),
            "accept_rate": 1.0,
        },
    }
    result = {
        "experiment": "EXP-006", "stage": "stage2_router_feasibility",
        "split": "train", "protocol_revision": payload["protocol_revision"],
        "source_features": args.features, "validation_accessed": False,
        "query_or_future_router_input": False,
        "model_selection": "none_fixed_pca16_ridge_alpha1",
        "candidate_utility_spearman": {
            **{
                name: float(spearmanr(prediction, targets).statistic)
                for name, prediction in predictions.items()
            },
            "current_objective_improvement": float(spearmanr(
                [row["current_objective_improvement"] for row in rows], targets,
            ).statistic),
        },
        "metrics": metrics, "folds": fold_rows,
        "rows": [{
            "fold": row["fold"], "episode": row["episode"], "candidate": row["candidate"],
            "future_utility": row["future_utility"],
            "current_objective_improvement": row["current_objective_improvement"],
            "ridge_full": float(prediction_full[index]),
            "ridge_online_scalars": float(prediction_online[index]),
            "ridge_consensus_mean": float(0.5 * (prediction_full[index] + prediction_online[index])),
            "ridge_conservative_min": float(min(prediction_full[index], prediction_online[index])),
            **{name: float(prediction[index]) for name, prediction in predictions.items()},
        } for index, row in enumerate(rows)],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "out": str(output), "spearman": result["candidate_utility_spearman"], "metrics": metrics,
    }), flush=True)


if __name__ == "__main__":
    main()
