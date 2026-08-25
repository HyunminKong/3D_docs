#!/usr/bin/env python3
"""Build causal utility pairs and fit one source-safe deployable address."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal, future_readout, track_objective
from revisit3d.models import PlasticityAtom, SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _cpu_atom,
    _device_atom,
    _identifier,
    _timestamp,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_features(current: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if current.shape != source.shape or current.numel() != 64:
        raise ValueError("unified address requires two 64-D descriptors")
    return torch.cat((current, source, current - source, current * source))


def _context_tables(manifest: list[dict]) -> tuple[dict, dict, dict]:
    context, targets, locations = {}, {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context")):
            segment = row[tag]
            key = _identifier(segment)
            info = {
                "id": key, "segment": segment, "cache_index": index, "cache_tag": cache_tag,
                "location": row["location"],
            }
            if key in context and context[key]["segment"] != segment:
                raise RuntimeError("duplicate context metadata changed")
            context.setdefault(key, info)
            if key in locations and locations[key] != row["location"]:
                raise RuntimeError("one context crosses official locations")
            locations[key] = row["location"]
        target_key = _identifier(row["a_prime"])
        target = {
            "id": target_key, "cache_index": index, "episode": f"target-{target_key}",
            "component": f"component-{int(row['component_id'])}", "location": row["location"],
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        }
        if target_key in targets and targets[target_key]["query_frames"] != target["query_frames"]:
            raise RuntimeError("duplicate target query changed")
        targets.setdefault(target_key, target)
    if len(context) != 557 or len(targets) != 218 or len(set(locations.values())) != 4:
        raise RuntimeError("EXP-016 context inventory contract failed")
    return context, targets, locations


def _build_pairs(config: dict, head: SpatialPlasticityHead, geometry: dict, manifest: list[dict], device):
    context, targets, locations = _context_tables(manifest)
    metadata_cache = {}
    scene_root = Path(config["data"]["scene_root"])
    for info in context.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)
    events = sorted(context.values(), key=lambda row: (row["timestamp"], row["id"]))
    memory: dict[str, dict] = {}
    bank: list[str] = []
    features, utility, metadata = [], [], []
    target_table = {
        target["episode"]: {"component": target["component"], "location": target["location"]}
        for target in targets.values()
    }
    panel_size = int(config["method"]["panel_size"])
    step_size = float(config["method"]["step_size"])
    strength = float(config["method"]["reuse_strength"])
    with torch.enable_grad():
        for event_index, event in enumerate(events):
            key = event["id"]
            payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
            role = "current" if key in targets else "source"
            segment = CachedAtomSegment.from_cache(payload, role, device)
            zero = segment.atom(head)
            code = adapt_minimal(head, segment, zero.code, step_size=step_size)
            atom = replace(zero, code=code.detach())
            state = {
                "atom": _cpu_atom(atom),
                "descriptor": zero.key.mean(dim=(1, 2))[0].detach().cpu(),
                "location": event["location"],
            }
            if key in targets and bank:
                target = targets[key]
                stable = int(hashlib.sha1(target["episode"].encode()).hexdigest()[:8], 16)
                generator = random.Random(int(config["seed"]) + stable)
                panel = generator.sample(bank, min(panel_size, len(bank)))
                query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                query = CachedAtomSegment.from_cache(query_payload, "query", device)
                query_zero = query.atom(head)
                current_query = future_readout(head, atom, query, query_zero)
                for candidate in panel:
                    source_state = memory[candidate]
                    source_atom = _device_atom(source_state["atom"], device)
                    transported = visual_transport(source_atom, zero).code
                    candidate_code = (code + strength * transported).clamp(-1, 1)
                    candidate_query = future_readout(
                        head, replace(zero, code=candidate_code), query, query_zero,
                    )
                    value = (
                        (current_query - candidate_query)
                        / current_query.detach().abs().clamp_min(1e-6)
                    ).detach()
                    features.append(_pair_features(state["descriptor"], source_state["descriptor"]))
                    utility.append(value.cpu())
                    metadata.append({
                        "episode": target["episode"], "component": target["component"],
                        "target_context": key, "source_context": candidate,
                        "target_location": target["location"],
                        "source_location": source_state["location"],
                    })
            memory[key] = state
            bank.append(key)
            if (event_index + 1) % 50 == 0 or event_index + 1 == len(events):
                print(json.dumps({
                    "events": event_index + 1, "total": len(events),
                    "targets": len({row['episode'] for row in metadata}), "pairs": len(metadata),
                }), flush=True)
    matrix = torch.stack(features).float()
    target = torch.stack(utility).float()
    if matrix.shape != (len(metadata), 256) or not torch.isfinite(matrix).all() or not torch.isfinite(target).all():
        raise RuntimeError("causal utility-pair tensor contract failed")
    return matrix, target, metadata, target_table


def _strict_oof(matrix, utility, target_location, source_location, alpha):
    prediction = np.full(len(utility), np.nan, dtype=np.float64)
    folds = []
    for held in sorted(set(target_location.tolist())):
        train = (target_location != held) & (source_location != held)
        test = target_location == held
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(matrix[train], utility[train])
        prediction[test] = model.predict(matrix[test])
        association = spearmanr(utility[test], prediction[test])
        folds.append({
            "held_out_location": held, "train_pairs": int(train.sum()), "test_pairs": int(test.sum()),
            "spearman": float(association.statistic), "spearman_p": float(association.pvalue),
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("strict source-safe OOF did not cover every pair")
    return prediction, folds


def _component_summary(values: dict[str, float], target_table: dict, accepted: dict[str, bool]) -> dict:
    groups = sorted({row["component"] for row in target_table.values()})
    per_group = {}
    for group in groups:
        episodes = [episode for episode, row in target_table.items() if row["component"] == group]
        per_group[group] = {
            "utility": float(np.mean([values.get(episode, 0.0) for episode in episodes])),
            "harm": float(np.mean([values.get(episode, 0.0) < 0 for episode in episodes])),
            "acceptance": float(np.mean([accepted.get(episode, False) for episode in episodes])),
        }
    return {
        "targets": len(target_table), "components": len(groups),
        "mean_utility": float(np.mean([row["utility"] for row in per_group.values()])),
        "harmful_rate": float(np.mean([row["harm"] for row in per_group.values()])),
        "acceptance": float(np.mean([row["acceptance"] for row in per_group.values()])),
    }


def _bootstrap(left, right, target_table, *, samples: int, seed: int):
    groups = sorted({row["component"] for row in target_table.values()})
    differences = np.asarray([
        np.mean([
            left.get(episode, 0.0) - right.get(episode, 0.0)
            for episode, row in target_table.items() if row["component"] == group
        ]) for group in groups
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(differences, size=(samples, len(groups)), replace=True).mean(axis=1)
    return {
        "unit": "physical_overlap_component", "components": len(groups),
        "mean_difference": float(differences.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _compile(model, matrix: np.ndarray) -> dict:
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    raw = ridge.coef_.astype(np.float64) / scaler.scale_.astype(np.float64)
    intercept = float(ridge.intercept_ - raw @ scaler.mean_.astype(np.float64))
    current, source, difference, product = np.split(raw, 4)
    compiled = {
        "current": current + difference,
        "source": source - difference,
        "interaction": product,
        "intercept": intercept,
    }
    probe = matrix[: min(256, len(matrix))]
    c, s = probe[:, :64], probe[:, 64:128]
    score = intercept + c @ compiled["current"] + np.sum(s * (compiled["source"] + c * product), axis=1)
    error = float(np.max(np.abs(score - model.predict(probe))))
    if error > 1e-10:
        raise RuntimeError(f"exact MIPS compilation error {error}")
    compiled["maximum_verification_error"] = error
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-016_unified_utility_address_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    cache_path = Path(config["output"]["candidate_cache"])
    artifact_path = Path(config["output"]["artifact"])
    if result_path.exists() or cache_path.exists() or artifact_path.exists():
        raise RuntimeError("EXP-016 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-016 requires train split and CUDA")
    atom_result = json.loads(Path(config["model"]["atom_result"]).read_text())
    checkpoint_path = Path(config["model"]["atom_checkpoint"])
    if not (
        atom_result["gate"]["passed"] is True
        and atom_result["checkpoint_sha256"] == _sha256(checkpoint_path)
        and atom_result["validation_accessed"] is False and atom_result["test_accessed"] is False
    ):
        raise RuntimeError("frozen EXP-015 atom contract failed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not (
        checkpoint["experiment"] == "EXP-015" and checkpoint["protocol_revision"] == "v1.0"
        and checkpoint["online_loss"] == "track3d_only" and checkpoint["auxiliary_losses"] == []
        and checkpoint["step_size"] == config["method"]["step_size"]
        and checkpoint["reuse_strength"] == config["method"]["reuse_strength"]
    ):
        raise RuntimeError("EXP-015 checkpoint fields changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    if len(manifest) != 225 or len(geometry.get("rows", [])) != 225 or geometry.get("split") != "train":
        raise RuntimeError("EXP-016 geometry cache contract failed")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["model"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    tensor, target, metadata, target_table = _build_pairs(config, head, geometry, manifest, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-016", "protocol_revision": config["protocol_revision"], "split": "train",
        "features": tensor, "utility": target, "metadata": metadata, "target_table": target_table,
        "feature_contract": "[current,source,current-source,current*source]",
        "query_or_future_online_input": False, "validation_accessed": False, "test_accessed": False,
    }, cache_path)
    matrix, utility = tensor.numpy().astype(np.float64), target.numpy().astype(np.float64)
    target_location = np.asarray([row["target_location"] for row in metadata])
    source_location = np.asarray([row["source_location"] for row in metadata])
    prediction, folds = _strict_oof(
        matrix, utility, target_location, source_location, float(config["address"]["ridge_alpha"]),
    )
    by_episode = {}
    for index, row in enumerate(metadata):
        by_episode.setdefault(row["episode"], []).append(index)
    policy, random_same_accept, appearance, oracle, accepted = {}, {}, {}, {}, {}
    threshold = float(config["method"]["acceptance_threshold"])
    for episode in target_table:
        indices = by_episode.get(episode, [])
        if not indices:
            policy[episode] = random_same_accept[episode] = appearance[episode] = oracle[episode] = 0.0
            accepted[episode] = False
            continue
        winner = max(indices, key=lambda index: (prediction[index], metadata[index]["source_context"]))
        take = bool(prediction[winner] > threshold)
        accepted[episode] = take
        policy[episode] = float(utility[winner]) if take else 0.0
        random_same_accept[episode] = float(np.mean(utility[indices])) if take else 0.0
        c = matrix[indices[0], :64]
        appearance_winner = max(
            indices,
            key=lambda index: float(
                c @ matrix[index, 64:128]
                / max(np.linalg.norm(c) * np.linalg.norm(matrix[index, 64:128]), 1e-12)
            ),
        )
        appearance[episode] = float(utility[appearance_winner]) if take else 0.0
        oracle[episode] = max(0.0, float(np.max(utility[indices])))
    policies = {
        "unified": _component_summary(policy, target_table, accepted),
        "random_same_accept": _component_summary(random_same_accept, target_table, accepted),
        "appearance_same_accept": _component_summary(appearance, target_table, accepted),
        "oracle_panel": _component_summary(oracle, target_table, {key: value > 0 for key, value in oracle.items()}),
    }
    bootstrap = {
        "unified_minus_random": _bootstrap(
            policy, random_same_accept, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "unified_minus_appearance": _bootstrap(
            policy, appearance, target_table,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    association = spearmanr(utility, prediction)
    success = config["success"]
    checks = {
        "positive_pooled_spearman": float(association.statistic) > 0,
        "positive_each_location_spearman": min(row["spearman"] for row in folds)
        > float(success["minimum_each_location_spearman"]),
        "minimum_policy_utility": policies["unified"]["mean_utility"]
        > float(success["minimum_policy_utility"]),
        "minimum_acceptance": policies["unified"]["acceptance"] >= float(success["minimum_acceptance"]),
        "maximum_harm": policies["unified"]["harmful_rate"] <= float(success["maximum_harmful_rate"]),
        "above_appearance": policies["unified"]["mean_utility"] > policies["appearance_same_accept"]["mean_utility"],
        "random_component_ci_positive": bootstrap["unified_minus_random"]["ci95"][0] > 0,
    }
    result = {
        "experiment": "EXP-016", "stage": "unified_utility_address", "split": "train",
        "protocol_revision": config["protocol_revision"], "config": str(config_path),
        "targets": len(target_table), "pairs": len(metadata), "components": 25, "locations": 4,
        "source_entity_excluded": True, "query_or_future_online_input": False,
        "candidate_cache": str(cache_path), "candidate_cache_sha256": _sha256(cache_path),
        "pair_oof_spearman": float(association.statistic), "pair_oof_spearman_p": float(association.pvalue),
        "folds": folds, "policies": policies, "bootstrap": bootstrap,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "validation_accessed": False, "test_accessed": False,
    }
    if all(checks.values()):
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(config["address"]["ridge_alpha"])))
        model.fit(matrix, utility)
        compiled = _compile(model, matrix)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "experiment": "EXP-016", "protocol_revision": config["protocol_revision"],
            "split": "train", "model": model, "compiled_mips": compiled,
            "feature_contract": "[current,source,current-source,current*source]",
            "descriptor_dim": 64, "topk": 1, "acceptance_threshold": threshold,
            "atom_checkpoint": str(checkpoint_path), "atom_checkpoint_sha256": _sha256(checkpoint_path),
            "validation_accessed": False, "test_accessed": False,
        }, artifact_path)
        result["artifact"] = str(artifact_path)
        result["artifact_sha256"] = _sha256(artifact_path)
        result["mips_maximum_verification_error"] = compiled["maximum_verification_error"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "pair_oof_spearman": result["pair_oof_spearman"],
        "folds": folds, "policies": policies, "bootstrap": bootstrap,
        "gate": result["registered_gate"], "artifact": result.get("artifact"),
    }), flush=True)
    if not all(checks.values()):
        raise RuntimeError(f"EXP-016 unified address gate failed: {checks}")


if __name__ == "__main__":
    main()
