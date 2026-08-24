#!/usr/bin/env python3
"""Reproduce EXP-006 v2.6 fold heads for out-of-fold diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.scripts.train_exp006_atom import _new_head, _train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument("--training-result", default="")
    parser.add_argument("--checkpoint-dir", default="revisit3d/checkpoints/EXP-006_crossfit_v26")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_crossfit_heads_train_v26.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 cross-fit training requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    result_path = Path(args.training_result or config["stage1"]["training_result"])
    registered = json.loads(result_path.read_text())
    if not (
        cache.get("protocol_revision") == registered.get("protocol_revision") == config["protocol_revision"]
        and cache.get("split") == registered.get("split") == "train"
    ):
        raise RuntimeError("cache/training-result protocol mismatch")
    steps = int(registered["selected_steps"])
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    records = dataset.records
    folds, group_of = grouped_folds(records, int(config["stage0"]["folds"]), int(config["seed"]))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = []
    for fold_index, held_out in enumerate(folds):
        train_indices = [index for index in range(len(records)) if index not in held_out]
        seed = int(config["seed"]) + fold_index
        head = _new_head(cache, config, device, seed)
        logs = _train(
            head, cache, records, train_indices, config, device,
            steps=steps, seed=seed, log_prefix=f"crossfit_fold_{fold_index}",
        )
        checkpoint_path = checkpoint_dir / f"fold_{fold_index}.pt"
        torch.save({
            "experiment": "EXP-006", "stage": "stage1_crossfit_head",
            "protocol_revision": config["protocol_revision"], "split": "train",
            "fold": fold_index, "held_out": held_out, "train_indices": train_indices,
            "steps": steps, "seed": seed, "head": head.state_dict(),
            "query_readout": "visual_only",
        }, checkpoint_path)
        rows.append({
            "fold": fold_index, "held_out": held_out,
            "held_out_groups": sorted({group_of[index] for index in held_out}),
            "train_indices": train_indices, "steps": steps, "seed": seed,
            "checkpoint": str(checkpoint_path),
            "last_train_outer": logs[-1]["outer"],
        })
    payload = {
        "experiment": "EXP-006", "stage": "stage1_crossfit_head_reproduction",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "source_training_result": str(result_path), "selected_steps": steps,
        "validation_accessed": False, "folds": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output), "folds": len(rows), "steps": steps}), flush=True)


if __name__ == "__main__":
    main()
