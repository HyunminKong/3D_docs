#!/usr/bin/env python3
"""Component-cross-fit and refit of the paper-minimal plasticity atom."""

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
from revisit3d.experiments.exp012_minimal import prepare_current, reuse_query_loss, run_minimal_episode
from revisit3d.models import SpatialPlasticityHead


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_head(cache: dict, config: dict, device: torch.device, seed: int) -> SpatialPlasticityHead:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = config["model"]
    head = SpatialPlasticityHead(
        feature_dim=int(model["feature_dim"]), key_dim=int(model["key_dim"]),
        code_dim=int(model["code_dim"]), hidden_dim=int(model["hidden_dim"]),
    ).to(device)
    head.initialize_key_projection(cache["pca_components"], cache["pca_mean"])
    head.key_projection.requires_grad_(False)
    return head


def _segments(cache: dict, index: int, device: torch.device):
    segments = cache["rows"][index]["segments"]
    return (
        CachedAtomSegment.from_cache(segments["a_context"], "source", device),
        CachedAtomSegment.from_cache(segments["b_context"], "source", device),
        CachedAtomSegment.from_cache(segments["a_prime_context"], "current", device),
        CachedAtomSegment.from_cache(segments["a_prime_query"], "query", device),
    )


def _component_folds(manifest: list[dict], folds: int, seed: int) -> tuple[list[list[int]], list[str]]:
    """Greedily balance immutable inventory component IDs without splitting one."""
    group_of = [f"component-{int(row['component_id'])}" for row in manifest]
    groups = sorted(set(group_of))
    if folds < 2 or folds > len(groups):
        raise ValueError("invalid number of component folds")
    generator = random.Random(seed)
    tie_break = groups.copy()
    generator.shuffle(tie_break)
    rank = {group: index for index, group in enumerate(tie_break)}
    size = {group: group_of.count(group) for group in groups}
    ordered = sorted(groups, key=lambda group: (-size[group], rank[group]))
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    fold_load = [0] * folds
    for group in ordered:
        destination = min(range(folds), key=lambda index: (fold_load[index], index))
        fold_groups[destination].append(group)
        fold_load[destination] += size[group]
    fold_indices = [
        [index for index, group in enumerate(group_of) if group in held]
        for held in fold_groups
    ]
    if sorted(index for fold in fold_indices for index in fold) != list(range(len(manifest))):
        raise RuntimeError("component folds do not partition the manifest")
    return fold_indices, group_of


def _train(
    head: SpatialPlasticityHead,
    cache: dict,
    indices: list[int],
    config: dict,
    device: torch.device,
    *,
    seed: int,
    label: str,
) -> list[dict]:
    training, method = config["training"], config["method"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
    )
    generator = random.Random(seed)
    logs = []
    head.train()
    step = 0
    for epoch in range(int(training["epochs"])):
        order = list(indices)
        generator.shuffle(order)
        for index in order:
            step += 1
            source, _, current, query = _segments(cache, index, device)
            optimizer.zero_grad(set_to_none=True)
            rollout = run_minimal_episode(
                head, source, current, query,
                step_size=float(method["step_size"]),
                reuse_strength=float(method["reuse_strength"]),
            )
            if not torch.isfinite(rollout.outer_loss):
                raise RuntimeError(f"non-finite EXP-012 outer loss at {label}:{step}")
            rollout.outer_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in head.parameters() if parameter.requires_grad],
                float(training["gradient_clip"]),
            )
            optimizer.step()
            if step == 1 or step % int(training["log_every"]) == 0 or step == len(indices) * int(training["epochs"]):
                row = {
                    "step": step, "epoch": epoch + 1,
                    "outer": float(rollout.outer_loss.detach()),
                    "current_to_base": float((rollout.current_query_loss / rollout.base_query_loss).detach()),
                    "matched_utility": float(rollout.utility), "gradient_norm": float(gradient),
                }
                logs.append(row)
                print(json.dumps({"phase": label, **row}), flush=True)
    return logs


def _evaluate(
    head: SpatialPlasticityHead,
    cache: dict,
    manifest: list[dict],
    indices: list[int],
    group_of: list[str],
    config: dict,
    device: torch.device,
) -> list[dict]:
    method = config["method"]
    rows = []
    head.eval()
    with torch.enable_grad():
        for index in indices:
            source_a, source_b, current, query = _segments(cache, index, device)
            state = prepare_current(head, current, query, step_size=float(method["step_size"]))
            losses = {}
            for name, source in (("matched", source_a), ("distant", source_b)):
                losses[name] = reuse_query_loss(
                    head, source, current, query, state,
                    step_size=float(method["step_size"]),
                    reuse_strength=float(method["reuse_strength"]),
                )
            denominator = state.current_query_loss.detach().abs().clamp_min(1e-6)
            utilities = {
                name: float(((state.current_query_loss - loss) / denominator).detach())
                for name, loss in losses.items()
            }
            rows.append({
                "index": index, "episode": manifest[index]["episode_id"], "group": group_of[index],
                "location": manifest[index]["location"],
                "current_to_base": float((state.current_query_loss / state.base_query_loss).detach()),
                "matched_utility": utilities["matched"], "distant_utility": utilities["distant"],
                "matched_harmful": utilities["matched"] < 0,
            })
    return rows


def _component_summary(rows: list[dict]) -> dict:
    groups = sorted({row["group"] for row in rows})
    keys = ("current_to_base", "matched_utility", "distant_utility")
    grouped = {
        group: {
            key: float(np.mean([row[key] for row in rows if row["group"] == group]))
            for key in keys
        } for group in groups
    }
    return {
        "rows": len(rows), "components": len(groups),
        **{key: float(np.mean([grouped[group][key] for group in groups])) for key in keys},
        "matched_harmful_rate": float(np.mean([row["matched_harmful"] for row in rows])),
    }


def _bootstrap_difference(rows: list[dict], *, samples: int, seed: int) -> dict:
    groups = sorted({row["group"] for row in rows})
    values = np.asarray([
        np.mean([
            row["matched_utility"] - row["distant_utility"]
            for row in rows if row["group"] == group
        ]) for group in groups
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": "matched_minus_distant", "components": len(values),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-012_minimal_atom_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError("EXP-012 Stage-0 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-012 Stage 0 requires train split and CUDA")
    prerequisite = json.loads(Path(config["prerequisite"]).read_text())
    if not (
        prerequisite["split"] == "val" and prerequisite["registered_gate"]["passed"] is True
        and prerequisite["locked_objective"]["name"] == "track3d_only"
        and prerequisite["locked_objective"]["step_size"] == config["method"]["step_size"]
        and prerequisite["test_accessed"] is False
    ):
        raise RuntimeError("EXP-011 prerequisite contract failed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train" and cache.get("protocol_revision") == "v1.5"
        and cache.get("pca_fit_split") == "train"
    ):
        raise RuntimeError("EXP-012 train cache contract failed")
    folds, group_of = _component_folds(
        manifest, int(config["crossfit"]["folds"]), int(config["seed"]),
    )
    if len(set(group_of)) != 25:
        raise RuntimeError("expected 25 physical overlap components")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    oof_rows, fold_logs = [], []
    for fold, held_out in enumerate(folds):
        train_indices = [index for index in range(len(manifest)) if index not in held_out]
        head = _new_head(cache, config, device, int(config["seed"]) + fold)
        logs = _train(
            head, cache, train_indices, config, device,
            seed=int(config["seed"]) + fold, label=f"fold-{fold}",
        )
        rows = _evaluate(head, cache, manifest, held_out, group_of, config, device)
        oof_rows.extend(rows)
        fold_logs.append({"fold": fold, "train": len(train_indices), "held_out": len(held_out), "logs": logs})
    if sorted(row["index"] for row in oof_rows) != list(range(len(manifest))):
        raise RuntimeError("OOF rows do not form an exact partition")
    summary = _component_summary(oof_rows)
    difference = _bootstrap_difference(
        oof_rows, samples=int(config["statistics"]["bootstrap_samples"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    success = config["success"]
    checks = {
        "current_improves_future_loss": summary["current_to_base"] < float(success["maximum_current_to_base"]),
        "matched_utility": summary["matched_utility"] > float(success["minimum_matched_utility"]),
        "matched_harm": summary["matched_harmful_rate"] <= float(success["maximum_matched_harm"]),
        "matched_beats_distant": difference["ci95"][0] > 0,
    }
    if not all(checks.values()):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "experiment": "EXP-012", "stage": "minimal_atom_crossfit", "split": "train",
            "protocol_revision": config["protocol_revision"], "summary": summary,
            "matched_minus_distant": difference, "gate": {"checks": checks, "passed": False},
            "oof_rows": oof_rows, "fold_logs": fold_logs,
            "validation_accessed": False, "test_accessed": False,
        }, indent=2, allow_nan=False))
        raise RuntimeError(f"EXP-012 minimal atom gate failed: {checks}")

    final_head = _new_head(cache, config, device, int(config["seed"]))
    final_logs = _train(
        final_head, cache, list(range(len(manifest))), config, device,
        seed=int(config["seed"]), label="full-train",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-012", "stage": "minimal_atom", "split": "train",
        "protocol_revision": config["protocol_revision"], "head": final_head.state_dict(),
        "frozen_visual_key": True, "online_loss": "track3d_only",
        "step_size": float(config["method"]["step_size"]),
        "reuse_strength": float(config["method"]["reuse_strength"]),
        "training_epochs": int(config["training"]["epochs"]),
        "validation_accessed": False, "test_accessed": False,
    }, checkpoint_path)
    result = {
        "experiment": "EXP-012", "stage": "minimal_atom_crossfit", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "method": config["method"], "model": config["model"],
        "meta_objective_terms": ["current_future_track3d", "matched_reuse_future_track3d"],
        "auxiliary_losses": [], "summary": summary, "matched_minus_distant": difference,
        "gate": {"checks": checks, "passed": True}, "oof_rows": oof_rows,
        "fold_logs": fold_logs, "final_logs": final_logs,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "validation_accessed": False, "test_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "checkpoint": str(checkpoint_path),
        "summary": summary, "matched_minus_distant": difference,
        "gate": result["gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
