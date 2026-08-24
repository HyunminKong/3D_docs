#!/usr/bin/env python3
"""Out-of-fold adaptation-step and memory-reuse budget comparison."""

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
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


def _summary(rows: list[dict], name: str, epsilon: float) -> dict:
    ratio = np.asarray([row[f"{name}_to_base"] for row in rows], dtype=np.float64)
    utility = np.asarray([row[f"{name}_vs_one_step_utility"] for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows), "mean_to_base": float(ratio.mean()),
        "median_to_base": float(np.median(ratio)),
        "mean_utility_vs_one_current_step": float(utility.mean()),
        "beneficial_rate_vs_one_step": float(np.mean(utility > epsilon)),
        "harmful_rate_vs_one_step": float(np.mean(utility < -epsilon)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--crossfit", default="revisit3d/results/EXP-006/stage1_crossfit_heads_train_v26.json",
    )
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_adaptation_budget_crossfit_train_v26.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 adaptation-budget diagnostic requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    crossfit = json.loads(Path(args.crossfit).read_text())
    if not (
        cache.get("protocol_revision") == crossfit.get("protocol_revision") == config["protocol_revision"]
        and crossfit.get("validation_accessed") is False
    ):
        raise RuntimeError("cache/cross-fit protocol mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    records = dataset.records
    device = torch.device("cuda")
    stage1 = config["stage1"]
    step_size = float(stage1["ttt_step_size"])
    strength = float(stage1["reuse_strength"])
    epsilon = float(stage1["utility_deadband_minimum"])
    rows = []
    seen: set[int] = set()
    for fold in crossfit["folds"]:
        checkpoint = torch.load(fold["checkpoint"], map_location="cpu", weights_only=False)
        head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
        head.load_state_dict(checkpoint["head"])
        head.eval().requires_grad_(False)
        with torch.enable_grad():
            for index in fold["held_out"]:
                if index in seen:
                    raise RuntimeError("cross-fit held-out duplication")
                seen.add(index)
                current, query, sources = _segments(cache, records, index, config, device)
                current_zero = current.atom(head)
                base_loss = query_readout_loss(head, current_zero, query)
                one_code, _ = adapt_context(
                    head, current, current_zero.code, step_size=step_size, steps=1,
                )
                one_loss = query_readout_loss(head, replace(current_zero, code=one_code), query)
                two_code, _ = adapt_context(head, current, one_code, step_size=step_size, steps=1)
                two_loss = query_readout_loss(head, replace(current_zero, code=two_code), query)
                transported = []
                for _, source in sources:
                    source_zero = source.atom(head)
                    source_code, _ = adapt_context(
                        head, source, source_zero.code, step_size=step_size, steps=1,
                    )
                    source_atom = replace(source_zero, code=source_code.detach())
                    transported.append(visual_transport(source_atom, current_zero).code)
                pooled = torch.stack(transported).mean(dim=0)
                memory_after_code = (one_code + strength * pooled).clamp(-1, 1)
                memory_after_loss = query_readout_loss(
                    head, replace(current_zero, code=memory_after_code), query,
                )
                memory_before_code, _ = adapt_context(
                    head, current, strength * pooled, step_size=step_size, steps=1,
                )
                memory_before_loss = query_readout_loss(
                    head, replace(current_zero, code=memory_before_code), query,
                )
                candidate_losses = []
                for code in transported:
                    candidate = (one_code + strength * code).clamp(-1, 1)
                    candidate_losses.append(query_readout_loss(
                        head, replace(current_zero, code=candidate), query,
                    ))
                oracle_loss = torch.minimum(one_loss, torch.stack(candidate_losses).min())
                losses = {
                    "one_current_step": one_loss,
                    "two_current_steps": two_loss,
                    "mean_memory_after_one_step": memory_after_loss,
                    "mean_memory_before_one_step": memory_before_loss,
                    "oracle_memory_after_one_step": oracle_loss,
                }
                row = {
                    "fold": fold["fold"], "index": index,
                    "episode": records[index]["episode_id"], "base_loss": float(base_loss.detach()),
                }
                for name, loss in losses.items():
                    row[f"{name}_loss"] = float(loss.detach())
                    row[f"{name}_to_base"] = float((loss / base_loss.detach().abs().clamp_min(1e-6)).detach())
                    row[f"{name}_vs_one_step_utility"] = float(
                        normalized_future_utility(one_loss, loss).detach()
                    )
                rows.append(row)
                print(json.dumps({"fold": fold["fold"], "episode": row["episode"], "done": True}), flush=True)
        del head
    if seen != set(range(len(records))):
        raise RuntimeError("cross-fit heads did not cover every train episode")
    names = (
        "one_current_step", "two_current_steps", "mean_memory_after_one_step",
        "mean_memory_before_one_step", "oracle_memory_after_one_step",
    )
    summary = {name: _summary(rows, name, epsilon) for name in names}
    payload = {
        "experiment": "EXP-006", "stage": "stage1_crossfit_adaptation_budget",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "crossfit_manifest": args.crossfit, "validation_accessed": False,
        "query_geometry_accessed": False, "reuse_strength": strength,
        "summary": summary, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
