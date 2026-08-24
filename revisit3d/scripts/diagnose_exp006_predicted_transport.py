#!/usr/bin/env python3
"""Train-only preflight for predicted Sim(3) atom alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import deterministic_foreign_indices, require_exp006_split
from revisit3d.models import SpatialPlasticityHead, align_atoms, build_plasticity_atom, geometry_transport


def _atom(head: SpatialPlasticityHead, segment: dict, device: torch.device):
    features = segment["features"].to(device=device, dtype=torch.float32)
    return build_plasticity_atom(
        head,
        features,
        segment["xyz"].to(device),
        segment["scale"].to(device),
        segment["base_confidence"].to(device=device, dtype=torch.float32),
        segment["track"].to(device=device, dtype=torch.float32),
        segment["track_visibility"].to(device=device, dtype=torch.float32),
        segment["track_confidence"].to(device=device, dtype=torch.float32),
        image_size=tuple(segment["image_size"]),
    )


def _record(alignment, label: str, candidate: str) -> dict:
    return {
        "label": label,
        "candidate": candidate,
        "valid": alignment.valid,
        "correspondences": alignment.correspondences,
        "inliers": alignment.inliers,
        "inlier_ratio": alignment.inlier_ratio,
        "normalized_median_residual": (
            alignment.normalized_median_residual if np.isfinite(alignment.normalized_median_residual) else None
        ),
        "sim3_scale": float(alignment.scale),
        "source_rank_ratio": alignment.source_rank_ratio,
        "target_rank_ratio": alignment.target_rank_ratio,
    }


def _summarize(rows: list[dict], label: str) -> dict:
    subset = [row for row in rows if row["label"] == label]
    valid = [row for row in subset if row["valid"]]
    return {
        "count": len(subset),
        "valid_rate": sum(row["valid"] for row in subset) / max(len(subset), 1),
        "median_correspondences": float(np.median([row["correspondences"] for row in subset])),
        "median_inlier_ratio": float(np.median([row["inlier_ratio"] for row in valid])) if valid else 0.0,
        "median_normalized_residual": float(np.median([
            row["normalized_median_residual"] for row in valid
        ])) if valid else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("predicted transport diagnostic requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    if cache.get("protocol_revision") != config["protocol_revision"] or cache.get("split") != "train":
        raise RuntimeError("Stage-1 cache protocol/split mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(config["data"]["image_height"], config["data"]["image_width"]),
    )
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=config["foundation"]["feature_dim"]).to(device)
    head.initialize_key_projection(cache["pca_components"], cache["pca_mean"])
    head.eval()
    rows = []
    with torch.no_grad():
        for index, cached in enumerate(cache["rows"]):
            current = _atom(head, cached["segments"]["a_prime_context"], device)
            query = _atom(head, cached["segments"]["a_prime_query"], device)
            source = _atom(head, cached["segments"]["a_context"], device)
            distractor = _atom(head, cached["segments"]["b_context"], device)
            pairs = [
                ("matched", cached["episode_id"], source),
                ("distant_b", cached["episode_id"], distractor),
            ]
            for foreign_index in deterministic_foreign_indices(dataset.records, index, 3, config["seed"]):
                foreign = cache["rows"][foreign_index]
                pairs.append(("foreign", foreign["episode_id"], _atom(
                    head, foreign["segments"]["a_context"], device,
                )))
            for label, candidate, atom in pairs:
                alignment = align_atoms(atom, current)[0]
                record = _record(alignment, label, candidate)
                if alignment.valid:
                    transported = geometry_transport(atom, current, [alignment], appearance_weight=5.0)
                    record.update({
                        "transport_finite": bool(torch.isfinite(transported.code).all()),
                        "transport_entropy": float(transported.normalized_entropy[0]),
                        "transport_coverage": float(transported.coverage[0]),
                    })
                else:
                    record.update({"transport_finite": False, "transport_entropy": None, "transport_coverage": 0.0})
                record["episode"] = cached["episode_id"]
                rows.append(record)
            query_alignment = align_atoms(current, query)[0]
            query_record = _record(query_alignment, "current_to_query", cached["episode_id"])
            query_record["episode"] = cached["episode_id"]
            query_record["transport_finite"] = query_alignment.valid
            query_record["transport_entropy"] = None
            query_record["transport_coverage"] = 0.0
            rows.append(query_record)
            print(json.dumps({
                "episode": cached["episode_id"],
                "matched_valid": rows[-6]["valid"],
                "query_valid": query_alignment.valid,
            }), flush=True)
    summary = {label: _summarize(rows, label) for label in (
        "matched", "distant_b", "foreign", "current_to_query",
    )}
    health = {
        "matched_valid_rate_ge_0.80": summary["matched"]["valid_rate"] >= 0.80,
        "current_to_query_valid_rate_ge_0.80": summary["current_to_query"]["valid_rate"] >= 0.80,
        "all_valid_transports_finite": all(row["transport_finite"] for row in rows if row["valid"]),
    }
    health["passed"] = all(health.values())
    payload = {
        "experiment": "EXP-006",
        "stage": "predicted_transport_train_preflight",
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "summary": summary,
        "health": health,
        "rows": rows,
    }
    output = Path(config["stage1"]["transport_diagnostic_result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"summary": summary, "health": health, "out": str(output)}))


if __name__ == "__main__":
    main()
