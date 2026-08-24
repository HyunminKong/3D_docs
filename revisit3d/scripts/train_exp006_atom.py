#!/usr/bin/env python3
"""Train EXP-006 spatial atoms with train-only grouped cross-validation."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    CachedAtomSegment,
    adapt_context,
    deterministic_foreign_indices,
    grouped_folds,
    query_readout_loss,
    require_exp006_split,
    run_episode,
)
from revisit3d.models import SpatialPlasticityHead


def _segments(cache: dict, records: list[dict], index: int, config: dict, device: torch.device):
    row = cache["rows"][index]
    current = CachedAtomSegment.from_cache(row["segments"]["a_prime_context"], "current", device)
    query = CachedAtomSegment.from_cache(row["segments"]["a_prime_query"], "query", device)
    sources = [
        ("matched_a", CachedAtomSegment.from_cache(row["segments"]["a_context"], "source", device)),
        ("distant_b", CachedAtomSegment.from_cache(row["segments"]["b_context"], "source", device)),
    ]
    foreign_indices = deterministic_foreign_indices(records, index, 3, int(config["seed"]))
    for rank, foreign_index in enumerate(foreign_indices):
        foreign = cache["rows"][foreign_index]
        sources.append((
            f"foreign_{rank}",
            CachedAtomSegment.from_cache(foreign["segments"]["a_context"], "source", device),
        ))
    if len(sources) != int(config["stage1"]["candidate_count"]):
        raise RuntimeError("candidate pool does not match the registered K")
    return current, query, sources


def _new_head(cache: dict, config: dict, device: torch.device, seed: int) -> SpatialPlasticityHead:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.initialize_key_projection(cache["pca_components"], cache["pca_mean"])
    return head


def _rollout(head, cache, records, index, config, device, *, build_outer=True):
    current, query, sources = _segments(cache, records, index, config, device)
    stage1 = config["stage1"]
    return run_episode(
        head, current, query, sources,
        step_size=float(stage1["ttt_step_size"]),
        ttt_steps=int(stage1["ttt_steps"]),
        appearance_weight=float(stage1["appearance_weight"]),
        reuse_strength=float(stage1["reuse_strength"]),
        utility_epsilon=float(stage1["utility_deadband_minimum"]),
        key_temperature=float(stage1["key_temperature"]),
        key_target_view=1 + index % (current.features.shape[1] - 1),
        transport_mode=stage1.get("transport_mode", "geometry_appearance"),
        build_outer=build_outer,
    )


def _train(
    head: SpatialPlasticityHead,
    cache: dict,
    records: list[dict],
    indices: list[int],
    config: dict,
    device: torch.device,
    *,
    steps: int,
    seed: int,
    log_prefix: str,
) -> list[dict]:
    stage1 = config["stage1"]
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(stage1["learning_rate"]),
        weight_decay=float(stage1["weight_decay"]),
    )
    generator = random.Random(seed)
    order: list[int] = []
    records_out = []
    head.train()
    for step in range(1, steps + 1):
        if not order:
            order = list(indices)
            generator.shuffle(order)
        index = order.pop()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rollout = _rollout(head, cache, records, index, config, device)
        if not torch.isfinite(rollout.outer_loss):
            raise RuntimeError(f"non-finite Stage-1 loss at {log_prefix} step {step}")
        rollout.outer_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), float(stage1["gradient_clip"]),
        )
        optimizer.step()
        row = {
            "step": step,
            "episode": records[index].get("episode_id", cache["rows"][index]["episode_id"]),
            "outer": float(rollout.outer_loss.detach()),
            "base_query": float(rollout.base_query_loss.detach()),
            "current_query": float(rollout.current_query_loss.detach()),
            "current_to_base": float(
                (rollout.current_query_loss / rollout.base_query_loss.detach().abs().clamp_min(1e-6)).detach()
            ),
            "mean_utility": float(rollout.utilities[rollout.valid].mean().detach()) if rollout.valid.any() else None,
            "best_utility": float(rollout.utilities[rollout.valid].max().detach()) if rollout.valid.any() else None,
            "valid_candidates": int(rollout.valid.sum()),
            "beneficial": rollout.counts["beneficial"],
            "neutral": rollout.counts["neutral"],
            "harmful": rollout.counts["harmful"],
            "key_loss": float(rollout.key_loss.detach()),
            "neutral_loss": float(rollout.neutral_loss.detach()),
            "center_loss": float(rollout.center_loss.detach()),
            "gradient_norm": float(gradient_norm),
            "wall_time_s": time.perf_counter() - started,
        }
        records_out.append(row)
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps({"phase": log_prefix, **row}), flush=True)
        del rollout
    return records_out


def _evaluate(
    head: SpatialPlasticityHead,
    cache: dict,
    records: list[dict],
    indices: list[int],
    group_of: list[str],
    config: dict,
    device: torch.device,
) -> dict:
    rows = []
    head.eval()
    with torch.enable_grad():
        for index in indices:
            rollout = _rollout(head, cache, records, index, config, device, build_outer=False)
            utility = rollout.utilities.detach().cpu()
            valid = rollout.valid.detach().cpu()
            valid_utility = utility[valid]
            best = max(0.0, float(valid_utility.max())) if valid_utility.numel() else 0.0
            row = {
                "index": index,
                "episode": records[index].get("episode_id", cache["rows"][index]["episode_id"]),
                "group": group_of[index],
                "current_query": float(rollout.current_query_loss.detach()),
                "base_query": float(rollout.base_query_loss.detach()),
                "current_to_base": float(
                    (rollout.current_query_loss / rollout.base_query_loss.detach().abs().clamp_min(1e-6)).detach()
                ),
                "best_valid_utility": best,
                "mean_valid_utility": float(valid_utility.mean()) if valid_utility.numel() else None,
                "valid_candidates": int(valid.sum()),
                "utilities": [float(value) for value in utility],
                "valid": [bool(value) for value in valid],
                "labels": [candidate.label for candidate in rollout.candidates],
            }
            rows.append(row)
            del rollout
    group_values = {}
    group_current = {}
    for group in sorted({row["group"] for row in rows}):
        group_values[group] = float(np.mean([
            row["best_valid_utility"] for row in rows if row["group"] == group
        ]))
        group_current[group] = float(np.mean([
            row["current_to_base"] for row in rows if row["group"] == group
        ]))
    return {
        "mean_group_best_valid_utility": float(np.mean(list(group_values.values()))) if group_values else 0.0,
        "mean_group_current_to_base": float(np.mean(list(group_current.values()))) if group_current else float("inf"),
        "group_best_valid_utility": group_values,
        "group_current_to_base": group_current,
        "rows": rows,
    }


def _calibrate_null(
    head: SpatialPlasticityHead,
    cache: dict,
    records: list[dict],
    config: dict,
    device: torch.device,
) -> dict:
    values = []
    head.eval()
    for index in range(len(records)):
        current, query, _ = _segments(cache, records, index, config, device)
        zero = current.atom(head)
        first_code, _ = adapt_context(
            head, current, zero.code, step_size=float(config["stage1"]["ttt_step_size"]),
            steps=int(config["stage1"]["ttt_steps"]),
        )
        first = query_readout_loss(head, replace(zero, code=first_code), query)
        second_code, _ = adapt_context(
            head, current, zero.code, step_size=float(config["stage1"]["ttt_step_size"]),
            steps=int(config["stage1"]["ttt_steps"]),
        )
        second = query_readout_loss(head, replace(zero, code=second_code), query)
        values.append(float(((first - second).abs() / first.detach().abs().clamp_min(1e-6)).detach()))
    percentile = float(np.percentile(values, 95))
    epsilon = max(float(config["stage1"]["utility_deadband_minimum"]), percentile)
    return {"absolute_null_utilities": values, "percentile_95": percentile, "epsilon": epsilon}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 atom training requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    if config["stage1"].get("query_readout") != "visual_only":
        raise RuntimeError("EXP-006 v2.4 requires visual-only future-query readout")
    if config["stage1"].get("reuse_application") != "additive_after_current":
        raise RuntimeError("EXP-006 v2.6 requires bounded residual reuse after current TTT")
    if config["protocol_revision"] == "v2.7" and config["stage1"].get("transport_mode") != "visual":
        raise RuntimeError("EXP-006 v2.7 requires visual fast-code transport")
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    if cache.get("protocol_revision") != config["protocol_revision"] or cache.get("split") != "train":
        raise RuntimeError("Stage-1 cache protocol/split mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    records = dataset.records
    if len(cache["rows"]) != len(records):
        raise RuntimeError("cache and manifest train rows differ")
    folds, group_of = grouped_folds(records, int(config["stage0"]["folds"]), int(config["seed"]))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    initial = _new_head(cache, config, device, int(config["seed"]))
    null = _calibrate_null(initial, cache, records, config, device)
    registered = float(config["stage1"]["utility_deadband_minimum"])
    if abs(null["epsilon"] - registered) > 1e-12:
        raise RuntimeError(
            f"train null control requires epsilon={null['epsilon']}; update config before training"
        )
    if args.smoke_steps:
        torch.cuda.reset_peak_memory_stats()
        logs = _train(
            initial, cache, records, list(range(len(records))), config, device,
            steps=args.smoke_steps, seed=int(config["seed"]), log_prefix="smoke",
        )
        payload = {
            "experiment": "EXP-006", "stage": "atom_smoke", "split": "train",
            "protocol_revision": config["protocol_revision"], "steps": args.smoke_steps,
            "null_control": null, "logs": logs,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
        print(json.dumps({"smoke": "passed", "summary": payload}), flush=True)
        return

    candidate_steps = [int(value) for value in config["stage1"]["steps_candidates"]]
    max_steps = max(candidate_steps)
    cv = []
    for fold_index, held_out in enumerate(folds):
        train_indices = [index for index in range(len(records)) if index not in held_out]
        head = _new_head(cache, config, device, int(config["seed"]) + fold_index)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=float(config["stage1"]["learning_rate"]),
            weight_decay=float(config["stage1"]["weight_decay"]),
        )
        generator = random.Random(int(config["seed"]) + fold_index)
        order: list[int] = []
        fold_logs = []
        evaluations = {}
        head.train()
        for step in range(1, max_steps + 1):
            if not order:
                order = list(train_indices)
                generator.shuffle(order)
            index = order.pop()
            optimizer.zero_grad(set_to_none=True)
            rollout = _rollout(head, cache, records, index, config, device)
            rollout.outer_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), float(config["stage1"]["gradient_clip"]),
            )
            optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"step": step, "outer": float(rollout.outer_loss.detach()),
                       "gradient_norm": float(gradient_norm), "valid_candidates": int(rollout.valid.sum())}
                fold_logs.append(row)
                print(json.dumps({"phase": f"cv_fold_{fold_index}", **row}), flush=True)
            del rollout
            if step in candidate_steps:
                evaluations[str(step)] = _evaluate(
                    head, cache, records, held_out, group_of, config, device,
                )
                head.train()
        cv.append({"fold": fold_index, "held_out": held_out, "logs": fold_logs, "evaluations": evaluations})
    scores = {
        step: {
            "mean_group_best_valid_utility": float(np.mean([
                fold["evaluations"][str(step)]["mean_group_best_valid_utility"] for fold in cv
            ])),
            "mean_group_current_to_base": float(np.mean([
                fold["evaluations"][str(step)]["mean_group_current_to_base"] for fold in cv
            ])),
        } for step in candidate_steps
    }
    maximum_current_ratio = float(config["stage1"]["maximum_current_to_base_ratio"])
    eligible_steps = [
        step for step in candidate_steps
        if scores[step]["mean_group_current_to_base"] <= maximum_current_ratio
    ]
    if not eligible_steps:
        raise RuntimeError(f"all Stage-1 checkpoints violate current/base guard: {scores}")
    selected_steps = max(
        eligible_steps,
        key=lambda step: (scores[step]["mean_group_best_valid_utility"], -step),
    )
    final_head = _new_head(cache, config, device, int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    final_logs = _train(
        final_head, cache, records, list(range(len(records))), config, device,
        steps=selected_steps, seed=int(config["seed"]), log_prefix="full_train_refit",
    )
    final_evaluation = _evaluate(
        final_head, cache, records, list(range(len(records))), group_of, config, device,
    )
    checkpoint_path = Path(config["stage1"]["output_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-006", "stage": 1, "protocol_revision": config["protocol_revision"],
        "split": "train", "head": final_head.state_dict(), "selected_steps": selected_steps,
        "cv_scores": scores, "null_control": null, "query_readout": "visual_only",
        "base_geometry_checkpoint": config["stage0"]["output_checkpoint"],
    }, checkpoint_path)
    result = {
        "experiment": "EXP-006", "stage": "atom_meta_training", "split": "train",
        "protocol_revision": config["protocol_revision"], "candidate_steps": candidate_steps,
        "selected_steps": selected_steps, "cv_scores": scores, "null_control": null,
        "cv": cv, "final_logs": final_logs, "final_train_evaluation": final_evaluation,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "checkpoint": str(checkpoint_path), "validation_accessed": False,
    }
    result_path = Path(config["stage1"]["training_result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "result": str(result_path), "checkpoint": str(checkpoint_path),
        "selected_steps": selected_steps, "cv_scores": scores,
        "final_train_utility": final_evaluation["mean_group_best_valid_utility"],
        "final_train_current_to_base": final_evaluation["mean_group_current_to_base"],
    }), flush=True)


if __name__ == "__main__":
    main()
