#!/usr/bin/env python3
"""Fit and freeze the D023 router using train-only OOF feature rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import PRIMARY_SCALAR_INDICES, primary_feature_columns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_validation_v28.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("protocol_revision") != "v2.8" or config["data"].get("split") != "val":
        raise RuntimeError("locked router fit requires the pre-validation v2.8 config")
    stage2 = config["stage2"]
    if tuple(stage2["primary_scalar_indices"]) != PRIMARY_SCALAR_INDICES:
        raise RuntimeError("config changed the D023 primary scalar contract")
    source_path = Path(stage2["train_features"])
    source = json.loads(source_path.read_text())
    contract = source.get("router_feature_contract", {})
    if not (
        source.get("split") == "train"
        and source.get("validation_accessed") is False
        and source.get("query_geometry_accessed") is False
        and contract.get("query_or_future_input") is False
        and contract.get("descriptor_dimensions") == stage2["descriptor_dimensions"]
        and contract.get("observable_scalar_dimensions") == stage2["observable_scalar_dimensions"]
        and contract.get("total_dimensions") == 280
    ):
        raise RuntimeError("router source violates the locked train-only observable contract")
    rows = source["router_features"]
    if len(rows) != 380 or len({row["episode"] for row in rows}) != 76:
        raise RuntimeError("locked router expects 380 candidates from 76 expanded-train episodes")
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    columns = primary_feature_columns(
        int(stage2["descriptor_dimensions"]), tuple(stage2["primary_scalar_indices"]),
    )
    if not np.isfinite(matrix).all() or not np.isfinite(targets).all():
        raise RuntimeError("non-finite train router input")
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=int(stage2["pca_components"]), random_state=int(stage2["pca_random_state"])),
        Ridge(alpha=float(stage2["ridge_alpha"])),
    )
    model.fit(matrix[:, columns], targets)
    model_path = Path(stage2["output_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "experiment": "EXP-006",
        "protocol_revision": "v2.8",
        "source_atom_protocol_revision": config["source_atom_protocol_revision"],
        "feature_columns": columns,
        "utility_threshold": float(stage2["utility_threshold"]),
        "model": model,
    }, model_path)
    prediction = model.predict(matrix[:, columns])
    result = {
        "experiment": "EXP-006",
        "stage": "stage2_locked_router_fit",
        "split": "train",
        "protocol_revision": "v2.8",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "source_features": str(source_path),
        "source_features_sha256": _sha256(source_path),
        "config": str(config_path),
        "feature_columns": columns,
        "excluded_alignment_scalar_indices": [12, 13, 14, 15],
        "train_candidates": len(rows),
        "train_episodes": len({row["episode"] for row in rows}),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "fit_diagnostic_only": {
            "mae": float(np.abs(prediction - targets).mean()),
            "target_mean": float(targets.mean()),
            "prediction_mean": float(prediction.mean()),
        },
        "evidence_note": "In-sample fit diagnostics are not validation or OOF evidence.",
    }
    result_path = Path(stage2["train_result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
