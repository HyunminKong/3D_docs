#!/usr/bin/env python3
"""Final core atom: current + absolute reuse + relative utility, no auxiliaries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from revisit3d.experiments.exp012_minimal import run_core_utility_episode
from revisit3d.scripts.train_exp012_minimal_atom import _component_folds, _new_head, _sha256
from revisit3d.scripts.train_exp012_utility_selected_atom import (
    _bootstrap,
    _episode_segments,
    _evaluate,
    _summary,
)


def _train_steps(head, cache, manifest, indices, config, device, *, seed: int, label: str) -> list[dict]:
    training, method = config["training"], config["method"]
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
    )
    generator = random.Random(seed)
    logs, order = [], []
    head.train()
    total_steps = int(training["steps"])
    for step in range(1, total_steps + 1):
        if not order:
            order = list(indices)
            generator.shuffle(order)
        index = order.pop()
        sources, labels, current, query = _episode_segments(
            cache, manifest, index, indices, config, device,
        )
        optimizer.zero_grad(set_to_none=True)
        rollout = run_core_utility_episode(
            head, sources, current, query,
            step_size=float(method["step_size"]), reuse_strength=float(method["reuse_strength"]),
        )
        if not torch.isfinite(rollout.outer_loss):
            raise RuntimeError(f"non-finite EXP-015 loss at {label}:{step}")
        rollout.outer_loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
        optimizer.step()
        if step == 1 or step % int(training["log_every"]) == 0 or step == total_steps:
            row = {
                "step": step, "outer": float(rollout.outer_loss.detach()),
                "current_to_base": float((rollout.current_query_loss / rollout.base_query_loss).detach()),
                "oracle_utility": float(rollout.utilities.max()),
                "mean_utility": float(rollout.utilities.mean()),
                "selected_label": labels[rollout.selected_index], "gradient_norm": float(gradient),
            }
            logs.append(row)
            print(json.dumps({"phase": label, **row}), flush=True)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-015_core_atom_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError("EXP-015 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-015 requires train split and CUDA")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior["experiment"] == "EXP-014" and prior["gate"]["passed"] is False
        and prior["gate"]["checks"]["oracle_utility"] is False
        and all(prior["gate"]["checks"][key] is True for key in (
            "current_improves_future_loss", "candidate_mean_utility",
            "candidate_harm", "selection_headroom",
        ))
        and prior["training_steps"] == 1000
        and prior["validation_accessed"] is False and prior["test_accessed"] is False
    ):
        raise RuntimeError("EXP-014 terminal contract changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train" and cache.get("protocol_revision") == "v1.5"
        and cache.get("pca_fit_split") == "train" and int(config["training"]["steps"]) == 1000
    ):
        raise RuntimeError("EXP-015 cache/budget contract failed")
    folds, group_of = _component_folds(
        manifest, int(config["crossfit"]["folds"]), int(config["seed"]),
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    oof_rows, fold_logs = [], []
    for fold, held_out in enumerate(folds):
        train_indices = [index for index in range(len(manifest)) if index not in held_out]
        head = _new_head(cache, config, device, int(config["seed"]) + fold)
        logs = _train_steps(
            head, cache, manifest, train_indices, config, device,
            seed=int(config["seed"]) + fold, label=f"core-fold-{fold}",
        )
        oof_rows.extend(_evaluate(head, cache, manifest, held_out, group_of, config, device))
        fold_logs.append({"fold": fold, "train": len(train_indices), "held_out": len(held_out), "logs": logs})
    if sorted(row["index"] for row in oof_rows) != list(range(len(manifest))):
        raise RuntimeError("EXP-015 OOF rows do not form an exact partition")
    summary = _summary(oof_rows)
    difference = _bootstrap(
        oof_rows, samples=int(config["statistics"]["bootstrap_samples"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    success = config["success"]
    checks = {
        "current_improves_future_loss": summary["current_to_base"] < float(success["maximum_current_to_base"]),
        "oracle_utility": summary["oracle_utility"] > float(success["minimum_oracle_utility"]),
        "candidate_mean_utility": summary["mean_utility"] > float(success["minimum_mean_utility"]),
        "candidate_harm": summary["candidate_harmful_rate"] <= float(success["maximum_candidate_harm"]),
        "selection_headroom": difference["ci95"][0] > 0,
    }
    base_result = {
        "experiment": "EXP-015", "stage": "core_atom_crossfit", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "training_steps": int(config["training"]["steps"]),
        "meta_objective": "current_plus_best_reuse_plus_relative_softplus",
        "meta_objective_weights": [], "auxiliary_losses": [], "visual_key": "frozen_train_pca",
        "summary": summary, "oracle_minus_mean": difference,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "oof_rows": oof_rows, "fold_logs": fold_logs,
        "validation_accessed": False, "test_accessed": False,
    }
    if not all(checks.values()):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(base_result, indent=2, allow_nan=False))
        raise RuntimeError(f"EXP-015 core atom gate failed: {checks}")
    final_head = _new_head(cache, config, device, int(config["seed"]))
    final_logs = _train_steps(
        final_head, cache, manifest, list(range(len(manifest))), config, device,
        seed=int(config["seed"]), label="core-full-train",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-015", "stage": "core_atom", "split": "train",
        "protocol_revision": config["protocol_revision"], "head": final_head.state_dict(),
        "visual_key": base_result["visual_key"], "online_loss": "track3d_only",
        "auxiliary_losses": [], "meta_objective": base_result["meta_objective"],
        "training_steps": int(config["training"]["steps"]),
        "step_size": float(config["method"]["step_size"]),
        "reuse_strength": float(config["method"]["reuse_strength"]),
        "validation_accessed": False, "test_accessed": False,
    }, checkpoint_path)
    result = {
        **base_result, "final_logs": final_logs, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "checkpoint": str(checkpoint_path),
        "summary": summary, "oracle_minus_mean": difference, "gate": result["gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
