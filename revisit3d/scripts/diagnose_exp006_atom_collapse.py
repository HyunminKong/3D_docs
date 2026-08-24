#!/usr/bin/env python3
"""Measure spatial and source selectivity of the trained EXP-006 atom."""

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
from revisit3d.experiments import adapt_context, require_exp006_split
from revisit3d.models import SpatialPlasticityHead, align_atoms, geometry_transport
from revisit3d.scripts.train_exp006_atom import _segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_atom_collapse_train_v25.json",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if not torch.cuda.is_available():
        raise SystemExit("atom collapse diagnostic requires CUDA")
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
    rows, global_codes = [], []
    stage1 = config["stage1"]
    for index, record in enumerate(dataset.records):
        current, _, sources = _segments(cache, dataset.records, index, config, device)
        current_zero = current.atom(head)
        for label, segment in sources:
            zero = segment.atom(head)
            code, _ = adapt_context(
                head, segment, zero.code, step_size=float(stage1["ttt_step_size"]),
                steps=int(stage1["ttt_steps"]),
            )
            atom = replace(zero, code=code.detach())
            residual = head.log_depth_residual(segment.features, atom.code)
            global_code = atom.code.mean(dim=(1, 2)).flatten()
            global_codes.append(global_code)
            alignment = align_atoms(atom.detach(), current_zero.detach())[0]
            if alignment.valid:
                transported = geometry_transport(
                    atom, current_zero, [alignment], appearance_weight=float(stage1["appearance_weight"]),
                ).code
                transported_std = float(transported.flatten(1, 2).std(dim=1).mean())
            else:
                transported_std = None
            rows.append({
                "episode": record.get("episode_id", cache["rows"][index]["episode_id"]),
                "candidate": label,
                "alignment_valid": alignment.valid,
                "global_code_norm": float(global_code.norm()),
                "code_token_std": float(atom.code.flatten(1, 2).std(dim=1).mean()),
                "code_token_rms": float(atom.code.square().mean().sqrt()),
                "residual_mean": float(residual.mean()),
                "residual_abs_mean": float(residual.abs().mean()),
                "residual_token_std": float(residual.flatten(1, 2).std(dim=1).mean()),
                "transported_code_token_std": transported_std,
            })
    matrix = torch.stack(global_codes)
    normalized = F.normalize(matrix, dim=-1)
    cosine = normalized @ normalized.transpose(0, 1)
    upper = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
    def mean(name: str) -> float:
        values = [row[name] for row in rows if row[name] is not None]
        return float(np.mean(values))
    summary = {
        "sources": len(rows),
        "pairwise_global_code_cosine_mean": float(upper.mean()),
        "pairwise_global_code_cosine_median": float(upper.median()),
        "global_code_across_source_std": float(matrix.std(dim=0).mean()),
        "global_code_norm_mean": mean("global_code_norm"),
        "code_token_std_mean": mean("code_token_std"),
        "code_token_rms_mean": mean("code_token_rms"),
        "residual_abs_mean": mean("residual_abs_mean"),
        "residual_token_std_mean": mean("residual_token_std"),
        "transported_code_token_std_mean": mean("transported_code_token_std"),
        "alignment_valid_rate": float(np.mean([row["alignment_valid"] for row in rows])),
    }
    payload = {
        "experiment": "EXP-006", "stage": "atom_collapse_diagnostic", "split": "train",
        "protocol_revision": config["protocol_revision"], "validation_accessed": False,
        "summary": summary, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
