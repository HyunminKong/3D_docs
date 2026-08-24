#!/usr/bin/env python3
"""Train-only oracle-pose diagnostic for the EXP-006 transport bottleneck.

This script never uses future-query geometry.  Supplied camera poses are used
only for the matched source A and current A' *context* frames to establish an
offline transport upper bound.  The deployable predicted-geometry path remains
unchanged and is evaluated beside that upper bound.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import adapt_context, query_readout_loss, require_exp006_split
from revisit3d.losses import normalized_future_utility
from revisit3d.models import (
    Sim3Alignment,
    SpatialPlasticityHead,
    align_atoms,
    apply_sim3,
    geometry_transport,
    visual_transport,
)
from revisit3d.scripts.train_exp006_atom import _segments


CONDITIONS = (
    "visual",
    "predicted_geometry",
    "predicted_geometry_appearance",
    "known_pose_geometry",
    "known_pose_geometry_appearance",
)


def _metric_per_predicted_unit(predicted_w2c: torch.Tensor, known_w2c: torch.Tensor) -> torch.Tensor:
    """Calibrate one segment's arbitrary VGGT translation/depth gauge."""
    known_relative = known_w2c @ torch.linalg.inv(known_w2c[:, :1])
    known_motion = known_relative[:, 1:, :3, 3].norm(dim=-1)
    predicted_motion = predicted_w2c[:, 1:, :3, 3].norm(dim=-1)
    usable = (known_motion > 1e-5) & (predicted_motion > 1e-5)
    if not usable.any():
        raise RuntimeError("context motion is insufficient to calibrate the pose gauge")
    return (known_motion[usable] / predicted_motion[usable]).median()


def _known_pose_alignment(
    source_predicted_w2c: torch.Tensor,
    target_predicted_w2c: torch.Tensor,
    source_known_w2c: torch.Tensor,
    target_known_w2c: torch.Tensor,
    *,
    correspondences: int,
) -> tuple[Sim3Alignment, torch.Tensor, torch.Tensor]:
    """Map source predicted-gauge points into the target predicted gauge.

    VGGT depth and pose translation share an arbitrary per-segment gauge, while
    nuScenes poses are metric.  Directly combining metric translation with
    VGGT depth is invalid.  Context-only relative camera motion estimates the
    metric-per-predicted-unit scale for each segment before composing the
    source-camera-0 to target-camera-0 Sim(3).
    """
    source_gauge = _metric_per_predicted_unit(source_predicted_w2c, source_known_w2c)
    target_gauge = _metric_per_predicted_unit(target_predicted_w2c, target_known_w2c)
    source_c2w0 = torch.linalg.inv(source_known_w2c[0, 0])
    target_w2c0 = target_known_w2c[0, 0]
    rotation = target_w2c0[:3, :3] @ source_c2w0[:3, :3]
    source_origin_in_target_metric = (
        target_w2c0[:3, :3] @ source_c2w0[:3, 3] + target_w2c0[:3, 3]
    )
    scale = source_gauge / target_gauge
    translation = source_origin_in_target_metric / target_gauge
    alignment = Sim3Alignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        valid=bool(
            torch.isfinite(scale).all()
            and torch.isfinite(rotation).all()
            and torch.isfinite(translation).all()
            and scale > 0
        ),
        correspondences=correspondences,
        inliers=correspondences,
        inlier_ratio=1.0,
        normalized_median_residual=0.0,
        source_rank_ratio=1.0,
        target_rank_ratio=1.0,
    )
    return alignment, source_gauge, target_gauge


def _neighbor_indices(source_xyz: torch.Tensor, target_xyz: torch.Tensor, k: int) -> torch.Tensor:
    distance = torch.cdist(target_xyz.float(), source_xyz.float())
    return distance.topk(k, dim=-1, largest=False).indices


def _neighbor_agreement(predicted: torch.Tensor, known: torch.Tensor) -> tuple[float, float]:
    top1 = float((predicted[:, 0] == known[:, 0]).float().mean())
    intersection = (predicted[:, :, None] == known[:, None, :]).any(dim=-1).float().sum(dim=-1)
    recall = float((intersection / known.shape[1]).mean())
    return top1, recall


def _summarize(rows: list[dict], condition: str, epsilon: float) -> dict:
    values = np.asarray([
        row["utility"] for row in rows if row["condition"] == condition and row["valid"]
    ], dtype=np.float64)
    total = sum(row["condition"] == condition for row in rows)
    return {
        "episodes": total,
        "valid_rate": float(values.size / max(total, 1)),
        "mean_utility": float(values.mean()) if values.size else None,
        "median_utility": float(np.median(values)) if values.size else None,
        "beneficial_rate": float(np.mean(values > epsilon)) if values.size else 0.0,
        "harmful_rate": float(np.mean(values < -epsilon)) if values.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out",
        default="revisit3d/results/EXP-006/stage1_oracle_transport_gap_train_v26.json",
    )
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 oracle transport diagnostic requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if config["data"]["split"] != "train":
        raise RuntimeError("oracle transport diagnosis is train-only")
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(config["stage1"]["output_checkpoint"], map_location="cpu", weights_only=False)
    if not (
        cache.get("protocol_revision") == checkpoint.get("protocol_revision") == config["protocol_revision"]
        and cache.get("split") == checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage-1 cache/checkpoint protocol mismatch")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    epsilon = float(stage1["utility_deadband_minimum"])
    strength = float(stage1["reuse_strength"])
    appearance_weight = float(stage1["appearance_weight"])
    rows: list[dict] = []
    alignments: list[dict] = []

    with torch.enable_grad():
        for index, sample in enumerate(dataset):
            current, query, sources = _segments(cache, dataset.records, index, config, device)
            label, source = sources[0]
            if label != "matched_a":
                raise RuntimeError("the first registered candidate must be matched_a")

            current_zero = current.atom(head)
            source_zero = source.atom(head)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            source_code, _ = adapt_context(
                head, source, source_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            source_atom = replace(source_zero, code=source_code.detach())
            current_query = query_readout_loss(head, replace(current_zero, code=current_code), query)

            # Known poses are accessed only for source/current contexts.  Query
            # camera tensors are intentionally neither read nor backprojected.
            source_w2c = sample["a"]["context"]["w2c"].unsqueeze(0).to(device)
            current_w2c = sample["a_prime"]["context"]["w2c"].unsqueeze(0).to(device)
            predicted_alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
            known_alignment, source_gauge, current_gauge = _known_pose_alignment(
                source.predicted_w2c,
                current.predicted_w2c,
                source_w2c,
                current_w2c,
                correspondences=source_atom.xyz.shape[1] * source_atom.xyz.shape[2],
            )

            transported = {
                "visual": visual_transport(source_atom, current_zero).code,
                "predicted_geometry": geometry_transport(
                    source_atom, current_zero, [predicted_alignment], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "predicted_geometry_appearance": geometry_transport(
                    source_atom, current_zero, [predicted_alignment],
                    appearance_weight=appearance_weight, neighbors=args.neighbors,
                ).code,
                "known_pose_geometry": geometry_transport(
                    source_atom, current_zero, [known_alignment], appearance_weight=0.0,
                    neighbors=args.neighbors,
                ).code,
                "known_pose_geometry_appearance": geometry_transport(
                    source_atom, current_zero, [known_alignment],
                    appearance_weight=appearance_weight, neighbors=args.neighbors,
                ).code,
            }
            for condition in CONDITIONS:
                valid = not condition.startswith("predicted_geometry") or predicted_alignment.valid
                if valid:
                    candidate_code = (current_code + strength * transported[condition]).clamp(-1, 1)
                    query_loss = query_readout_loss(
                        head, replace(current_zero, code=candidate_code), query,
                    )
                    utility = normalized_future_utility(current_query, query_loss)
                    query_value = float(query_loss.detach())
                    utility_value = float(utility.detach())
                else:
                    query_value = None
                    utility_value = None
                rows.append({
                    "episode": sample["episode_id"],
                    "condition": condition,
                    "valid": bool(valid),
                    "query_loss": query_value,
                    "utility": utility_value,
                })

            neighbor = {
                "episode": sample["episode_id"],
                "predicted_alignment_valid": predicted_alignment.valid,
                "predicted_alignment_inlier_ratio": predicted_alignment.inlier_ratio,
                "predicted_alignment_residual": (
                    predicted_alignment.normalized_median_residual if predicted_alignment.valid else None
                ),
                "known_pose_alignment_valid": known_alignment.valid,
                "source_metric_per_predicted_unit": float(source_gauge),
                "current_metric_per_predicted_unit": float(current_gauge),
                "known_pose_alignment_scale": float(known_alignment.scale),
            }
            if predicted_alignment.valid:
                predicted_source_xyz = apply_sim3(
                    source_atom.xyz[0].flatten(0, 1), predicted_alignment,
                )
                predicted_target_xyz = current_zero.xyz[0].flatten(0, 1)
                known_source_xyz = apply_sim3(
                    source_atom.xyz[0].flatten(0, 1), known_alignment,
                )
                known_target_xyz = current_zero.xyz[0].flatten(0, 1)
                predicted_neighbors = _neighbor_indices(
                    predicted_source_xyz, predicted_target_xyz, args.neighbors,
                )
                known_neighbors = _neighbor_indices(
                    known_source_xyz, known_target_xyz, args.neighbors,
                )
                top1, recall = _neighbor_agreement(predicted_neighbors, known_neighbors)
                neighbor["predicted_vs_known_top1_agreement"] = top1
                neighbor[f"predicted_vs_known_recall_at_{args.neighbors}"] = recall
            else:
                neighbor["predicted_vs_known_top1_agreement"] = None
                neighbor[f"predicted_vs_known_recall_at_{args.neighbors}"] = None
            alignments.append(neighbor)
            print(json.dumps({"episode": sample["episode_id"], **neighbor}), flush=True)

    summary = {condition: _summarize(rows, condition, epsilon) for condition in CONDITIONS}
    valid_alignment = [row for row in alignments if row["predicted_alignment_valid"]]
    agreement_summary = {
        "predicted_alignment_valid_rate": len(valid_alignment) / max(len(alignments), 1),
        "mean_top1_agreement": float(np.mean([
            row["predicted_vs_known_top1_agreement"] for row in valid_alignment
        ])) if valid_alignment else None,
        f"mean_recall_at_{args.neighbors}": float(np.mean([
            row[f"predicted_vs_known_recall_at_{args.neighbors}"] for row in valid_alignment
        ])) if valid_alignment else None,
    }
    payload = {
        "experiment": "EXP-006",
        "stage": "stage1_oracle_transport_gap_diagnostic",
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "known_pose_diagnostic_only": True,
        "known_pose_scope": "matched_source_and_current_context_pose_gauge_only",
        "pose_gauge_calibration": "context_relative_motion_only",
        "query_geometry_accessed": False,
        "validation_accessed": False,
        "neighbors": args.neighbors,
        "reuse_application": config["stage1"]["reuse_application"],
        "reuse_strength": strength,
        "summary": summary,
        "neighbor_agreement": agreement_summary,
        "alignments": alignments,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary, **agreement_summary}), flush=True)


if __name__ == "__main__":
    main()
