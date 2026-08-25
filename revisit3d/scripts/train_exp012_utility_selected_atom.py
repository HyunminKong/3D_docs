#!/usr/bin/env python3
"""Utility-selected, one-objective atom refit after matched identity failed."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import run_utility_selected_episode
from revisit3d.scripts.train_exp012_minimal_atom import _component_folds, _new_head, _sha256


def _foreign_indices(
    manifest: list[dict], current_index: int, pool: list[int], count: int, seed: int,
) -> list[int]:
    current = manifest[current_index]
    current_scenes = {current["source_scene"], current["target_scene"]}
    eligible = [
        index for index in pool
        if index != current_index
        and not current_scenes.intersection({manifest[index]["source_scene"], manifest[index]["target_scene"]})
    ]
    if len(eligible) < count:
        raise RuntimeError("not enough component-safe foreign sources")
    eligible.sort(key=lambda index: manifest[index]["episode_id"])
    token = f"{seed}:{current['episode_id']}".encode()
    offset = int(hashlib.sha256(token).hexdigest()[:8], 16) % len(eligible)
    rotated = eligible[offset:] + eligible[:offset]
    return rotated[:count]


def _episode_segments(
    cache: dict,
    manifest: list[dict],
    index: int,
    pool: list[int],
    config: dict,
    device: torch.device,
):
    segments = cache["rows"][index]["segments"]
    sources = [
        CachedAtomSegment.from_cache(segments["a_context"], "source", device),
        CachedAtomSegment.from_cache(segments["b_context"], "source", device),
    ]
    labels = ["matched", "distant"]
    foreign = _foreign_indices(
        manifest, index, pool, int(config["meta_candidates"]["foreign_count"]), int(config["seed"]),
    )
    for rank, foreign_index in enumerate(foreign):
        foreign_segment = cache["rows"][foreign_index]["segments"]["a_context"]
        sources.append(CachedAtomSegment.from_cache(foreign_segment, "source", device))
        labels.append(f"foreign-{rank}")
    if len(sources) != int(config["meta_candidates"]["count"]):
        raise RuntimeError("registered meta-candidate count changed")
    current = CachedAtomSegment.from_cache(segments["a_prime_context"], "current", device)
    query = CachedAtomSegment.from_cache(segments["a_prime_query"], "query", device)
    return sources, labels, current, query


def _train(head, cache, manifest, indices, config, device, *, seed: int, label: str) -> list[dict]:
    training, method = config["training"], config["method"]
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
    )
    generator = random.Random(seed)
    logs, step = [], 0
    head.train()
    for epoch in range(int(training["epochs"])):
        order = list(indices)
        generator.shuffle(order)
        for index in order:
            step += 1
            sources, labels, current, query = _episode_segments(
                cache, manifest, index, indices, config, device,
            )
            optimizer.zero_grad(set_to_none=True)
            rollout = run_utility_selected_episode(
                head, sources, current, query,
                step_size=float(method["step_size"]), reuse_strength=float(method["reuse_strength"]),
            )
            if not torch.isfinite(rollout.outer_loss):
                raise RuntimeError(f"non-finite Stage-0B loss at {label}:{step}")
            rollout.outer_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip"]))
            optimizer.step()
            if step == 1 or step % int(training["log_every"]) == 0 or step == len(indices) * int(training["epochs"]):
                row = {
                    "step": step, "epoch": epoch + 1, "outer": float(rollout.outer_loss.detach()),
                    "current_to_base": float((rollout.current_query_loss / rollout.base_query_loss).detach()),
                    "oracle_utility": float(rollout.utilities.max()),
                    "mean_utility": float(rollout.utilities.mean()),
                    "selected_label": labels[rollout.selected_index], "gradient_norm": float(gradient),
                }
                logs.append(row)
                print(json.dumps({"phase": label, **row}), flush=True)
    return logs


def _evaluate(head, cache, manifest, indices, group_of, config, device) -> list[dict]:
    method = config["method"]
    rows = []
    head.eval()
    with torch.enable_grad():
        for index in indices:
            sources, labels, current, query = _episode_segments(
                cache, manifest, index, indices, config, device,
            )
            rollout = run_utility_selected_episode(
                head, sources, current, query,
                step_size=float(method["step_size"]), reuse_strength=float(method["reuse_strength"]),
            )
            utility = rollout.utilities.detach().cpu().numpy()
            rows.append({
                "index": index, "episode": manifest[index]["episode_id"], "group": group_of[index],
                "location": manifest[index]["location"],
                "current_to_base": float((rollout.current_query_loss / rollout.base_query_loss).detach()),
                "oracle_utility": float(utility.max()), "mean_utility": float(utility.mean()),
                "candidate_harmful_rate": float((utility < 0).mean()),
                "utilities": [float(value) for value in utility], "labels": labels,
            })
    return rows


def _summary(rows: list[dict]) -> dict:
    groups = sorted({row["group"] for row in rows})
    metrics = ("current_to_base", "oracle_utility", "mean_utility", "candidate_harmful_rate")
    component = {
        group: {
            metric: float(np.mean([row[metric] for row in rows if row["group"] == group]))
            for metric in metrics
        } for group in groups
    }
    return {
        "rows": len(rows), "components": len(groups),
        **{metric: float(np.mean([component[group][metric] for group in groups])) for metric in metrics},
    }


def _bootstrap(rows: list[dict], *, samples: int, seed: int) -> dict:
    groups = sorted({row["group"] for row in rows})
    values = np.asarray([
        np.mean([
            row["oracle_utility"] - row["mean_utility"]
            for row in rows if row["group"] == group
        ]) for group in groups
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": "oracle_minus_candidate_mean", "components": len(values),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-012_utility_selected_atom_v11.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError("EXP-012 Stage-0B output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-012 Stage 0B requires train split and CUDA")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior["protocol_revision"] == "v1.0" and prior["gate"]["passed"] is False
        and prior["gate"]["checks"]["matched_utility"] is False
        and prior["gate"]["checks"]["matched_beats_distant"] is False
        and prior["validation_accessed"] is False and prior["test_accessed"] is False
    ):
        raise RuntimeError("Stage-0A failure contract changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train" and cache.get("protocol_revision") == "v1.5"
        and cache.get("pca_fit_split") == "train"
    ):
        raise RuntimeError("EXP-012 Stage-0B cache contract failed")
    folds, group_of = _component_folds(
        manifest, int(config["crossfit"]["folds"]), int(config["seed"]),
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    oof_rows, fold_logs = [], []
    for fold, held_out in enumerate(folds):
        train_indices = [index for index in range(len(manifest)) if index not in held_out]
        head = _new_head(cache, config, device, int(config["seed"]) + fold)
        logs = _train(
            head, cache, manifest, train_indices, config, device,
            seed=int(config["seed"]) + fold, label=f"utility-fold-{fold}",
        )
        oof_rows.extend(_evaluate(head, cache, manifest, held_out, group_of, config, device))
        fold_logs.append({"fold": fold, "train": len(train_indices), "held_out": len(held_out), "logs": logs})
    if sorted(row["index"] for row in oof_rows) != list(range(len(manifest))):
        raise RuntimeError("Stage-0B OOF rows do not form an exact partition")
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
        "experiment": "EXP-012", "stage": "utility_selected_atom_crossfit", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "meta_objective": "equal_mean_current_and_min_candidate_future_track3d",
        "auxiliary_losses": [], "summary": summary, "oracle_minus_mean": difference,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "oof_rows": oof_rows, "fold_logs": fold_logs,
        "validation_accessed": False, "test_accessed": False,
    }
    if not all(checks.values()):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(base_result, indent=2, allow_nan=False))
        raise RuntimeError(f"EXP-012 utility-selected gate failed: {checks}")
    final_head = _new_head(cache, config, device, int(config["seed"]))
    final_logs = _train(
        final_head, cache, manifest, list(range(len(manifest))), config, device,
        seed=int(config["seed"]), label="utility-full-train",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-012", "stage": "utility_selected_minimal_atom", "split": "train",
        "protocol_revision": config["protocol_revision"], "head": final_head.state_dict(),
        "frozen_visual_key": True, "online_loss": "track3d_only", "auxiliary_losses": [],
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
