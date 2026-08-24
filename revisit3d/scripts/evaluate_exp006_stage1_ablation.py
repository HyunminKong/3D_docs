#!/usr/bin/env python3
"""Train-only A0--A5 atom/transport ablation before EXP-006 Stage 2."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    adapt_context,
    query_readout_loss,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import (
    SpatialPlasticityHead,
    align_atoms,
    geometry_transport,
    visual_transport,
)
from revisit3d.scripts.train_exp006_atom import _segments


CONDITIONS = (
    "global_vector", "untransported_local", "visual", "geometry", "geometry_appearance",
)


def _summarize(rows: list[dict], condition: str, epsilon: float) -> dict:
    subset = [row for row in rows if row["condition"] == condition]
    valid = [row for row in subset if row["valid"]]
    utilities = np.asarray([row["utility"] for row in valid], dtype=np.float64)
    per_episode = []
    for episode in sorted({row["episode"] for row in subset}):
        values = [row["utility"] for row in valid if row["episode"] == episode]
        per_episode.append(max([0.0, *values]))
    return {
        "candidates": len(subset),
        "valid_rate": len(valid) / max(len(subset), 1),
        "mean_utility": float(utilities.mean()) if utilities.size else None,
        "median_utility": float(np.median(utilities)) if utilities.size else None,
        "beneficial_rate": float(np.mean(utilities > epsilon)) if utilities.size else 0.0,
        "harmful_rate": float(np.mean(utilities < -epsilon)) if utilities.size else 0.0,
        "episodes_with_beneficial": sum(value > epsilon for value in per_episode),
        "mean_episode_best_utility": float(np.mean(per_episode)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_transport_ablation_train_v25.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 atom ablation requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(config["stage1"]["output_checkpoint"], map_location="cpu", weights_only=False)
    if not (
        cache.get("protocol_revision") == checkpoint.get("protocol_revision") == config["protocol_revision"]
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage-1 cache/checkpoint protocol mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    records = dataset.records
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    epsilon = float(stage1["utility_deadband_minimum"])
    rows = []
    episode_current = []
    with torch.enable_grad():
        for index in range(len(records)):
            current, query, sources = _segments(cache, records, index, config, device)
            current_zero = current.atom(head)
            base_query = query_readout_loss(head, current_zero, query)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            current_atom = replace(current_zero, code=current_code)
            current_query = query_readout_loss(head, current_atom, query)
            episode_current.append({
                "episode": records[index].get("episode_id", cache["rows"][index]["episode_id"]),
                "base_query": float(base_query.detach()),
                "current_query": float(current_query.detach()),
                "current_to_base": float((current_query / base_query.abs().clamp_min(1e-6)).detach()),
            })
            for label, source_segment in sources:
                source_zero = source_segment.atom(head)
                source_code, _ = adapt_context(
                    head, source_segment, source_zero.code,
                    step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
                )
                source_atom = replace(source_zero, code=source_code.detach())
                alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
                transported = {
                    "global_vector": source_code.detach().mean(dim=(1, 2), keepdim=True).expand_as(current_code),
                    "untransported_local": source_code.detach(),
                    "visual": visual_transport(source_atom, current_zero).code,
                    "geometry": geometry_transport(source_atom, current_zero, [alignment], appearance_weight=0.0).code,
                    "geometry_appearance": geometry_transport(
                        source_atom, current_zero, [alignment],
                        appearance_weight=float(stage1["appearance_weight"]),
                    ).code,
                }
                for condition in CONDITIONS:
                    valid = condition not in ("geometry", "geometry_appearance") or alignment.valid
                    if valid:
                        code = (
                            current_code + float(stage1["reuse_strength"]) * transported[condition]
                        ).clamp(-1, 1)
                        loss = query_readout_loss(head, replace(current_zero, code=code), query)
                        utility = normalized_future_utility(current_query, loss)
                        loss_value, utility_value = float(loss.detach()), float(utility.detach())
                    else:
                        loss_value, utility_value = None, None
                    rows.append({
                        "episode": episode_current[-1]["episode"], "candidate": label,
                        "condition": condition, "valid": bool(valid),
                        "query_loss": loss_value, "utility": utility_value,
                        "alignment_inlier_ratio": alignment.inlier_ratio,
                        "alignment_residual": alignment.normalized_median_residual if alignment.valid else None,
                    })
            print(json.dumps({"episode": episode_current[-1]["episode"], "done": True}), flush=True)
    summary = {condition: _summarize(rows, condition, epsilon) for condition in CONDITIONS}
    payload = {
        "experiment": "EXP-006", "stage": "stage1_train_ablation", "split": "train",
        "protocol_revision": config["protocol_revision"], "utility_epsilon": epsilon,
        "query_readout": "visual_only", "validation_accessed": False,
        "mean_current_to_base": float(np.mean([row["current_to_base"] for row in episode_current])),
        "summary": summary, "episode_current": episode_current, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
