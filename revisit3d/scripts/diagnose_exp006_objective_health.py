#!/usr/bin/env python3
"""Check whether learned current TTT improves an offline known-pose objective."""

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
from revisit3d.losses import track_3d_consistency_loss
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


def _known_loss(depth, query, known_w2c):
    side = int(depth.shape[2] ** 0.5)
    grid = depth.squeeze(-1).reshape(depth.shape[0], depth.shape[1], side, side)
    return track_3d_consistency_loss(
        grid, query.intrinsics, known_w2c, query.track,
        query.track_visibility, query.track_confidence, image_size=query.image_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_objective_health_train_v25.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if not torch.cuda.is_available():
        raise SystemExit("objective-health diagnostic requires CUDA")
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    checkpoint = torch.load(config["stage1"]["output_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_revision") != config["protocol_revision"]:
        raise RuntimeError("Stage-1 checkpoint revision mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    rows = []
    for index, sample in enumerate(dataset):
        current, query, _ = _segments(cache, dataset.records, index, config, device)
        current_zero = current.atom(head)
        current_code, _ = adapt_context(
            head, current, current_zero.code, step_size=float(stage1["ttt_step_size"]),
            steps=int(stage1["ttt_steps"]),
        )
        current_atom = replace(current_zero, code=current_code)
        query_atom = query.atom(head)
        query_code = visual_transport(current_atom, query_atom).code
        base_code = torch.zeros_like(query_code)
        base_depth = head.depth(query.features, query.base_depth, base_code)
        current_depth = head.depth(query.features, query.base_depth, query_code)
        known_w2c = sample["a_prime"]["query"]["w2c"].unsqueeze(0).to(device)
        predicted_base = query_readout_loss(head, current_zero, query)
        predicted_current = query_readout_loss(head, current_atom, query)
        known_base = _known_loss(base_depth, query, known_w2c)
        known_current = _known_loss(current_depth, query, known_w2c)
        log_ratio = torch.log(current_depth / base_depth.clamp_min(1e-6))
        rows.append({
            "episode": sample["episode_id"],
            "predicted_pose_base_loss": float(predicted_base.detach()),
            "predicted_pose_current_loss": float(predicted_current.detach()),
            "predicted_pose_current_to_base": float((predicted_current / predicted_base).detach()),
            "known_pose_base_loss": float(known_base.detach()),
            "known_pose_current_loss": float(known_current.detach()),
            "known_pose_current_to_base": float((known_current / known_base).detach()),
            "depth_ratio_mean": float((current_depth / base_depth.clamp_min(1e-6)).mean()),
            "depth_log_ratio_abs_mean": float(log_ratio.abs().mean()),
            "depth_log_ratio_std": float(log_ratio.std()),
        })
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "predicted_pose_current_to_base_mean": float(np.mean([
            row["predicted_pose_current_to_base"] for row in rows
        ])),
        "known_pose_current_to_base_mean": float(np.mean([
            row["known_pose_current_to_base"] for row in rows
        ])),
        "known_pose_improved_rate": float(np.mean([
            row["known_pose_current_to_base"] < 1 for row in rows
        ])),
        "depth_ratio_mean": float(np.mean([row["depth_ratio_mean"] for row in rows])),
        "depth_log_ratio_abs_mean": float(np.mean([row["depth_log_ratio_abs_mean"] for row in rows])),
        "depth_log_ratio_std_mean": float(np.mean([row["depth_log_ratio_std"] for row in rows])),
    }
    payload = {
        "experiment": "EXP-006", "stage": "objective_health_diagnostic", "split": "train",
        "protocol_revision": config["protocol_revision"], "known_pose_diagnostic_only": True,
        "validation_accessed": False, "summary": summary, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
