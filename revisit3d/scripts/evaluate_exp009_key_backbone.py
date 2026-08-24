#!/usr/bin/env python3
"""Leave-one-location-out comparison of frozen consolidation representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.scripts.simulate_exp007_token_bucket import _token_pair_features


def _identifier(segment: dict) -> str:
    payload = f"{segment['scene']}:{','.join(map(str, segment['frames']))}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_key_backbone_v14.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError(f"EXP-009 key result already exists: {output}")
    pairs = json.loads(Path(config["data"]["pair_manifest"]).read_text())
    caches = {
        name: torch.load(
            config["output"][f"{name}_feature_cache"], map_location="cpu", weights_only=False,
        ) for name in ("vggt", "dinov2")
    }
    if not all(
        cache.get("split") == "train"
        and cache.get("validation_accessed") is False
        and cache.get("test_accessed") is False
        and cache.get("protocol_revision") == config["protocol_revision"]
        and cache.get("model") == name
        for name, cache in caches.items()
    ):
        raise RuntimeError("EXP-009 key evaluation requires both locked train-only caches")
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    locations = np.asarray([row["location"] for row in pairs])
    if set(labels.tolist()) != {0, 1} or len(set(locations.tolist())) != 4:
        raise RuntimeError("pilot labels/locations are unhealthy")

    representations = []
    for name in ("vggt", "dinov2"):
        cache = caches[name]
        feature_rows = []
        pooled = []
        for pair in pairs:
            left = cache["rows"][_identifier(pair["left"])][name].float().numpy()
            right = cache["rows"][_identifier(pair["right"])][name].float().numpy()
            feature_rows.append(_token_pair_features(left, right))
            left_mean = left.mean(0)
            right_mean = right.mean(0)
            pooled.append(float(left_mean @ right_mean / max(
                np.linalg.norm(left_mean) * np.linalg.norm(right_mean), 1e-12,
            )))
        matrix = np.asarray(feature_rows, dtype=np.float64)
        pooled = np.asarray(pooled, dtype=np.float64)
        probability = np.empty(len(pairs), dtype=np.float64)
        folds = []
        for held_out in sorted(set(locations.tolist())):
            train, test = locations != held_out, locations == held_out
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(config["classifier"]["regularization_c"]),
                    class_weight=config["classifier"]["class_weight"],
                    max_iter=2000, random_state=int(config["seed"]),
                ),
            )
            model.fit(matrix[train], labels[train])
            probability[test] = model.predict_proba(matrix[test])[:, 1]
            prediction = probability[test] >= float(config["classifier"]["probability_threshold"])
            folds.append({
                "held_out_location": held_out, "pairs": int(test.sum()),
                "positives": int(labels[test].sum()),
                "roc_auc": float(roc_auc_score(labels[test], probability[test])),
                "balanced_accuracy": float(balanced_accuracy_score(labels[test], prediction)),
                "pooled_cosine_auc": float(roc_auc_score(labels[test], pooled[test])),
            })
        result = {
            "representation": name,
            "oof_roc_auc": float(roc_auc_score(labels, probability)),
            "oof_balanced_accuracy": float(balanced_accuracy_score(
                labels, probability >= float(config["classifier"]["probability_threshold"]),
            )),
            "pooled_cosine_auc": float(roc_auc_score(labels, pooled)),
            "pooled_cosine_spearman": float(spearmanr(labels, pooled).statistic),
            "folds": folds,
        }
        representations.append(result)
        print(json.dumps(result), flush=True)

    by_name = {row["representation"]: row for row in representations}
    dino = by_name["dinov2"]
    vggt = by_name["vggt"]
    margin = dino["oof_roc_auc"] - vggt["oof_roc_auc"]
    select_dino = (
        margin >= float(config["selection"]["minimum_dinov2_margin_over_vggt"])
        and min(row["roc_auc"] for row in dino["folds"])
        >= float(config["selection"]["minimum_dinov2_each_location_auc"])
    )
    selected = "dinov2" if select_dino else config["selection"]["fallback"]
    result = {
        "experiment": "EXP-009", "stage": "stage4_train_key_backbone_selection",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "config": str(config_path), "pairs": len(pairs),
        "representations": representations,
        "dinov2_margin_over_vggt": margin,
        "selected_consolidation_representation": selected,
        "registered_gate": {"dinov2_selected": bool(select_dino)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "selected": selected, "margin": margin,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
