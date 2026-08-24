#!/usr/bin/env python3
"""Fit and hash the train-only utility-MIPS address and final utility router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import PRIMARY_SCALAR_INDICES, primary_feature_columns
from revisit3d.scripts.evaluate_exp009_nested_router import (
    _calibrate_threshold,
    _episode_winners,
    _model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compile_mips(model) -> dict:
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    effective = np.asarray(ridge.coef_, dtype=np.float64) / np.asarray(scaler.scale_)
    intercept = float(ridge.intercept_ - effective @ np.asarray(scaler.mean_))
    current, source, difference, product = np.split(effective, 4)
    return {
        "intercept": intercept,
        "current_weight": current + difference,
        "source_weight": source - difference,
        "product_weight": product,
    }


def _mips_score(compiled: dict, matrix: np.ndarray) -> np.ndarray:
    current, source, _, _ = np.split(matrix, 4, axis=1)
    return (
        compiled["intercept"]
        + current @ compiled["current_weight"]
        + np.sum(source * (compiled["source_weight"] + current * compiled["product_weight"]), axis=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_validation_lock_v22.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    artifact_path = Path(config["output"]["artifact"])
    result_path = Path(config["output"]["train_lock_result"])
    if artifact_path.exists() or result_path.exists():
        raise RuntimeError("EXP-009 deployable lock output already exists")
    stage10 = json.loads(Path(config["source"]["stage10_result"]).read_text())
    stage11 = json.loads(Path(config["source"]["stage11_result"]).read_text())
    prefilter = json.loads(Path(config["source"]["stage8_candidate_cache"]).read_text())
    router_rows = json.loads(Path(config["source"]["stage5_candidate_cache"]).read_text())
    router_result = json.loads(Path(config["source"]["stage5_result"]).read_text())
    if not (
        stage10.get("registered_gate", {}).get("passed") is True
        and stage10.get("split") == stage11.get("split") == prefilter.get("split")
        == router_rows.get("split") == router_result.get("split") == "train"
        and all(payload.get("validation_accessed") is False for payload in (
            stage10, stage11, prefilter, router_rows, router_result,
        ))
        and all(payload.get("test_accessed") is False for payload in (
            stage10, stage11, prefilter, router_rows, router_result,
        ))
        and all(payload.get("query_or_future_router_input") is False for payload in (
            stage10, stage11, prefilter, router_rows, router_result,
        ))
    ):
        raise RuntimeError("deployable artifact requires locked train-only sources")

    prefilter_matrix = np.asarray([
        row["prefilter_features"][8:264] for row in prefilter["rows"]
    ], dtype=np.float64)
    prefilter_utility = np.asarray([
        row["future_utility"] for row in prefilter["rows"]
    ], dtype=np.float64)
    address = make_pipeline(StandardScaler(), Ridge(alpha=float(config["address"]["ridge_alpha"])))
    address.fit(prefilter_matrix, prefilter_utility)
    compiled = _compile_mips(address)
    exact_error = float(np.max(np.abs(
        address.predict(prefilter_matrix) - _mips_score(compiled, prefilter_matrix)
    )))
    if exact_error > 1e-10:
        raise RuntimeError(f"compiled utility-MIPS is not exact: {exact_error}")

    rows = router_rows["rows"]
    full_matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["component"] for row in rows])
    columns = primary_feature_columns(
        int(config["router"]["descriptor_dimensions"]),
        tuple(config["router"]["primary_scalar_indices"]),
    )
    if tuple(config["router"]["primary_scalar_indices"]) != PRIMARY_SCALAR_INDICES:
        raise RuntimeError("final router feature contract changed")
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    for held_out in sorted(set(groups.tolist())):
        train, test = groups != held_out, groups == held_out
        model = _model(config)
        model.fit(full_matrix[train][:, columns], utility[train])
        oof[test] = model.predict(full_matrix[test][:, columns])
    if not np.isfinite(oof).all():
        raise RuntimeError("final router OOF calibration is incomplete")
    visual_by_episode = {
        row["episode"]: float(row["visual_mean_utility"])
        for row in router_result["selection_rows"]
    }
    calibration = _calibrate_threshold(
        _episode_winners(rows, oof, np.arange(len(rows))), visual_by_episode,
        float(config["stage1"]["utility_deadband_minimum"]),
        float(config["router"]["minimum_acceptance"]),
    )
    router = _model(config)
    router.fit(full_matrix[:, columns], utility)
    payload = {
        "experiment": "EXP-009", "protocol_revision": config["protocol_revision"],
        "split": "train", "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False,
        "address_model": address, "address_feature_columns": [8, 264],
        "utility_mips": compiled, "mips_dimensions": 64,
        "router_model": router, "router_feature_columns": columns,
        "router_threshold": float(calibration["threshold"]),
        "router_calibration": calibration,
        "bank_capacity": int(config["bank"]["capacity"]),
        "bank_retention": config["bank"]["retention"],
        "candidate_count": int(config["bank"]["candidate_count"]),
        "reuse_strength": float(config["stage1"]["reuse_strength"]),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)
    result = {
        "experiment": "EXP-009", "stage": "stage12_deployable_train_lock",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "artifact": str(artifact_path), "artifact_sha256": _sha256(artifact_path),
        "address_train_pairs": len(prefilter_matrix),
        "router_train_candidates": len(rows),
        "mips_dimensions": 64, "mips_max_absolute_error": exact_error,
        "router_calibration": calibration,
        "selected_bank": "deterministic_reservoir_capacity8",
        "source_safe_address_evidence": stage10["variants"]["transport_descriptor"],
        "capacity_train_evidence": stage11["metrics"]["reservoir_capacity8"],
        "validation_accessed_after_lock": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
