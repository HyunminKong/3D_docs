#!/usr/bin/env python3
"""Train-only reuse-strength and interference diagnostic for EXP-006 atoms."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import adapt_context, query_readout_loss, require_exp006_split
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, geometry_transport, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_reuse_strength_train_v25.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if not torch.cuda.is_available():
        raise SystemExit("reuse-strength diagnostic requires CUDA")
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
    strengths = (0.05, 0.10, 0.25, 0.50, 1.0)
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
            transported = {
                "visual": visual_transport(source_atom, current_zero).code,
                "geometry_appearance": geometry_transport(
                    source_atom, current_zero, [alignment],
                    appearance_weight=float(stage1["appearance_weight"]),
                ).code,
            }
            for transport, raw in transported.items():
                valid = transport == "visual" or alignment.valid
                if not valid:
                    continue
                agreement = float(F.cosine_similarity(
                    raw.flatten(1), current_code.flatten(1), dim=-1,
                )[0])
                centered = raw - raw.mean(dim=(1, 2), keepdim=True)
                for strength in strengths:
                    initial, _ = adapt_context(
                        head, current, strength * raw,
                        step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
                    )
                    variants = {
                        "initialization": initial,
                        "additive": (current_code + strength * raw).clamp(-1, 1),
                        "additive_centered": (current_code + strength * centered).clamp(-1, 1),
                    }
                    for mode, code in variants.items():
                        loss = query_readout_loss(head, replace(current_zero, code=code), query)
                        rows.append({
                            "episode": record.get("episode_id", cache["rows"][index]["episode_id"]),
                            "candidate": label, "transport": transport, "mode": mode,
                            "strength": strength, "agreement": agreement,
                            "utility": float(normalized_future_utility(current_loss, loss).detach()),
                        })
        print(json.dumps({"episode": record.get("episode_id", index), "done": True}), flush=True)
    epsilon = float(stage1["utility_deadband_minimum"])
    summary = {}
    keys = sorted({(row["transport"], row["mode"], row["strength"]) for row in rows})
    for transport, mode, strength in keys:
        subset = [row for row in rows if (
            row["transport"], row["mode"], row["strength"]
        ) == (transport, mode, strength)]
        utility = np.asarray([row["utility"] for row in subset])
        agreement = np.asarray([row["agreement"] for row in subset])
        positive = utility[agreement > 0]
        negative = utility[agreement <= 0]
        name = f"{transport}_{mode}_a{strength:.2f}"
        summary[name] = {
            "count": len(subset), "mean_utility": float(utility.mean()),
            "median_utility": float(np.median(utility)),
            "beneficial_rate": float(np.mean(utility > epsilon)),
            "harmful_rate": float(np.mean(utility < -epsilon)),
            "agreement_positive_rate": float(np.mean(agreement > 0)),
            "positive_agreement_mean_utility": float(positive.mean()) if positive.size else None,
            "negative_agreement_mean_utility": float(negative.mean()) if negative.size else None,
        }
    payload = {
        "experiment": "EXP-006", "stage": "reuse_strength_diagnostic", "split": "train",
        "protocol_revision": config["protocol_revision"], "validation_accessed": False,
        "summary": summary, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    best = sorted(summary.items(), key=lambda item: item[1]["mean_utility"], reverse=True)[:10]
    print(json.dumps({"out": str(output), "best": best}), flush=True)


if __name__ == "__main__":
    main()
