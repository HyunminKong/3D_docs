#!/usr/bin/env python3
"""Train-only sensitivity diagnostic for Stage-1 transport smoothing."""

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
from revisit3d.models import SpatialPlasticityHead, align_atoms, geometry_transport, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_transport_kernel_train_v25.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if not torch.cuda.is_available():
        raise SystemExit("transport-kernel diagnostic requires CUDA")
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
    conditions = {
        **{f"visual_t{temperature}": ("visual", temperature) for temperature in (0.01, 0.03, 0.07)},
        **{f"geometry_k{neighbors}": ("geometry", neighbors, 0.0) for neighbors in (1, 2, 4, 8, 16)},
        **{f"geometry_appearance_k{neighbors}": ("geometry", neighbors, 5.0) for neighbors in (1, 2, 4, 8, 16)},
    }
    rows = []
    for index, record in enumerate(dataset.records):
        current, query, sources = _segments(cache, dataset.records, index, config, device)
        current_zero = current.atom(head)
        current_code, _ = adapt_context(
            head, current, current_zero.code, step_size=float(stage1["ttt_step_size"]),
            steps=int(stage1["ttt_steps"]),
        )
        current_loss = query_readout_loss(head, replace(current_zero, code=current_code), query)
        for label, segment in sources:
            source_zero = segment.atom(head)
            source_code, _ = adapt_context(
                head, segment, source_zero.code, step_size=float(stage1["ttt_step_size"]),
                steps=int(stage1["ttt_steps"]),
            )
            source_atom = replace(source_zero, code=source_code.detach())
            alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
            for name, specification in conditions.items():
                if specification[0] == "visual":
                    result = visual_transport(source_atom, current_zero, temperature=specification[1])
                    valid = True
                else:
                    result = geometry_transport(
                        source_atom, current_zero, [alignment], neighbors=specification[1],
                        appearance_weight=specification[2],
                    )
                    valid = alignment.valid
                if valid:
                    code, _ = adapt_context(
                        head, current, result.code, step_size=float(stage1["ttt_step_size"]),
                        steps=int(stage1["ttt_steps"]),
                    )
                    loss = query_readout_loss(head, replace(current_zero, code=code), query)
                    utility = float(normalized_future_utility(current_loss, loss).detach())
                    spatial_std = float(result.code.flatten(1, 2).std(dim=1).mean())
                else:
                    utility, spatial_std = None, None
                rows.append({
                    "episode": record.get("episode_id", cache["rows"][index]["episode_id"]),
                    "candidate": label, "condition": name, "valid": bool(valid),
                    "utility": utility, "transported_code_std": spatial_std,
                })
        print(json.dumps({"episode": record.get("episode_id", index), "done": True}), flush=True)
    epsilon = float(stage1["utility_deadband_minimum"])
    summary = {}
    for name in conditions:
        subset = [row for row in rows if row["condition"] == name]
        valid = [row for row in subset if row["valid"]]
        utility = np.asarray([row["utility"] for row in valid])
        episode_best = []
        for episode in sorted({row["episode"] for row in subset}):
            values = [row["utility"] for row in valid if row["episode"] == episode]
            episode_best.append(max([0.0, *values]))
        summary[name] = {
            "valid_rate": len(valid) / len(subset),
            "mean_utility": float(utility.mean()),
            "median_utility": float(np.median(utility)),
            "beneficial_rate": float(np.mean(utility > epsilon)),
            "harmful_rate": float(np.mean(utility < -epsilon)),
            "mean_episode_best_utility": float(np.mean(episode_best)),
            "transported_code_std": float(np.mean([row["transported_code_std"] for row in valid])),
        }
    payload = {
        "experiment": "EXP-006", "stage": "transport_kernel_diagnostic", "split": "train",
        "protocol_revision": config["protocol_revision"], "validation_accessed": False,
        "summary": summary, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
