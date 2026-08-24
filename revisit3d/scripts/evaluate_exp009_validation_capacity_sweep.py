#!/usr/bin/env python3
"""Validation-only capacity selection for the locked EXP-009 utility-MIPS bank."""

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

from revisit3d.experiments import (
    CachedAtomSegment,
    adapt_context,
    geometry_objective,
    observable_router_features,
    query_readout_loss,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.evaluate_exp006_validation import _summarize
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _cpu_atom,
    _device_atom,
    _float_stats,
    _identifier,
    _paired_component_bootstrap,
    _tensor_stats,
    _timestamp,
)
from revisit3d.scripts.evaluate_exp009_locked_validation import _mips_score, _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_capacity_selection_v23.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 validation capacity selection requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    validation_config = yaml.safe_load(Path(config["source"]["validation_config"]).read_text())
    result_path = Path(config["output"]["result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 Stage-14 output already exists")
    locked = json.loads(Path(config["source"]["locked_validation_result"]).read_text())
    old_candidates = json.loads(Path(config["source"]["locked_validation_candidates"]).read_text())
    if not (
        locked.get("split") == old_candidates.get("split") == "val"
        and locked.get("validation_accessed") is True
        and old_candidates.get("validation_accessed") is True
        and locked.get("test_accessed") is False
        and old_candidates.get("test_accessed") is False
        and locked.get("query_or_future_router_input") is False
        and old_candidates.get("query_or_future_router_input") is False
        and locked.get("registered_gate", {}).get("passed") is False
    ):
        raise RuntimeError("Stage 14 requires the preserved failed validation result")
    old_by_pair = {
        (row["episode"], row["candidate_context"]): row for row in old_candidates["rows"]
    }

    artifact_path = Path(validation_config["output"]["artifact"])
    lock = json.loads(Path(validation_config["output"]["train_lock_result"]).read_text())
    artifact = joblib.load(artifact_path)
    if not (
        lock["artifact_sha256"] == locked["artifact_sha256"] == _sha256(artifact_path)
        and artifact.get("protocol_revision") == "v2.2"
    ):
        raise RuntimeError("locked model artifact changed during capacity selection")
    manifest = json.loads(Path(validation_config["data"]["manifest"]).read_text())
    geometry = torch.load(
        validation_config["stage1"]["cache"], map_location="cpu", weights_only=False, mmap=True,
    )
    if not (len(manifest) == len(geometry.get("rows", [])) == 117):
        raise RuntimeError("validation manifest/cache changed")

    context_info, targets = {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (
            ("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context"),
        ):
            segment = row[tag]
            key = _identifier(segment)
            context_info.setdefault(key, {
                "id": key, "segment": segment, "cache_index": index, "cache_tag": cache_tag,
                "location": row["location"],
            })
        key = _identifier(row["a_prime"])
        targets.setdefault(key, {
            "id": key, "cache_index": index, "episode": f"target-{key}",
            "component": f"component-{int(row['component_id'])}",
            "location": row["location"],
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        })
    if len(context_info) != 241 or len(targets) != 103:
        raise RuntimeError("Stage-14 validation context contract changed")
    group_by_episode = {target["episode"]: target["component"] for target in targets.values()}
    metadata_cache = {}
    scene_root = Path(validation_config["data"]["scene_root"])
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)

    checkpoint = torch.load(
        validation_config["stage1"]["source_checkpoint"], map_location="cpu", weights_only=False,
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(validation_config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    capacities = [int(value) for value in config["capacity_selection"]["candidates"]]
    candidate_count = int(config["capacity_selection"]["candidate_count"])
    strength = float(validation_config["stage1"]["reuse_strength"])
    epsilon = float(config["statistics"]["utility_deadband"])
    router = artifact["router_model"]
    router_columns = artifact["router_feature_columns"]
    threshold = float(artifact["router_threshold"])
    compiled = artifact["utility_mips"]
    policies = ["unbounded"] + [f"reservoir_{cap}" for cap in capacities] + [f"fifo_{cap}" for cap in capacities]
    values = {policy: {} for policy in policies}
    accepted = {policy: {} for policy in policies}
    random_pools = {cap: {} for cap in capacities}
    new_rows, reused = [], 0

    with torch.enable_grad():
        for location in sorted({row["location"] for row in context_info.values()}):
            events = sorted(
                [row for row in context_info.values() if row["location"] == location],
                key=lambda row: (row["timestamp"], row["id"]),
            )
            memory = {}
            banks = {policy: [] for policy in policies}
            seen = 0
            generators = {
                cap: random.Random(
                    int(config["seed"]) + cap * 1009
                    + int(hashlib.sha1(location.encode()).hexdigest()[:8], 16)
                ) for cap in capacities
            }
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code,
                    step_size=float(validation_config["stage1"]["ttt_step_size"]),
                    steps=int(validation_config["stage1"]["ttt_steps"]),
                )
                pre, pre_stats = geometry_objective(head, segment, zero.code, return_stats=True)
                post, post_stats = geometry_objective(head, segment, code, return_stats=True)
                state = {
                    "atom": _cpu_atom(replace(zero, code=code.detach())),
                    "descriptor": zero.key.mean(dim=(1, 2))[0].detach().cpu(),
                    "pre": float(pre.detach()), "post": float(post.detach()),
                    "pre_stats": _float_stats(pre_stats), "post_stats": _float_stats(post_stats),
                }
                if key in targets:
                    target = targets[key]
                    episode = target["episode"]
                    candidate_sets, address_scores = {}, {}
                    for policy in policies:
                        ranked = sorted((
                            (candidate, _mips_score(
                                compiled, state["descriptor"], memory[candidate]["descriptor"],
                            )) for candidate in banks[policy]
                        ), key=lambda row: (-row[1], row[0]))[:candidate_count]
                        candidate_sets[policy] = [row[0] for row in ranked]
                        address_scores[policy] = dict(ranked)
                    union = {candidate for subset in candidate_sets.values() for candidate in subset}
                    for cap in capacities:
                        union.update(banks[f"reservoir_{cap}"])
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    current_query = query_readout_loss(head, replace(zero, code=code), query)
                    evaluated = {}
                    for candidate in sorted(union):
                        previous = old_by_pair.get((episode, candidate))
                        if previous is not None:
                            utility = float(previous["future_utility"])
                            prediction = float(previous["predicted_utility"])
                            reused += 1
                        else:
                            source_state = memory[candidate]
                            source_atom = _device_atom(source_state["atom"], device)
                            alignment = align_atoms(source_atom.detach(), zero.detach())[0]
                            visual = visual_transport(source_atom, zero)
                            candidate_code = (code + strength * visual.code).clamp(-1, 1)
                            candidate_objective = geometry_objective(head, segment, candidate_code)
                            candidate_query = query_readout_loss(
                                head, replace(zero, code=candidate_code), query,
                            )
                            features = observable_router_features(
                                current_descriptor=zero.key.mean(dim=(1, 2))[0],
                                source_descriptor=source_atom.key.mean(dim=(1, 2))[0],
                                current_code=code, transported_code=visual.code,
                                visual_result=visual, alignment=alignment,
                                current_pre_objective=pre, current_post_objective=post,
                                candidate_objective=candidate_objective,
                                source_pre_objective=torch.tensor(source_state["pre"], device=device),
                                source_post_objective=torch.tensor(source_state["post"], device=device),
                                current_pre_stats=pre_stats, current_post_stats=post_stats,
                                source_pre_stats=_tensor_stats(source_state["pre_stats"], device),
                                source_post_stats=_tensor_stats(source_state["post_stats"], device),
                            )
                            utility = float(normalized_future_utility(current_query, candidate_query).detach())
                            prediction = float(router.predict(
                                np.asarray(features.detach().cpu(), dtype=np.float64)[None, router_columns]
                            )[0])
                        evaluated[candidate] = {"utility": utility, "prediction": prediction}
                        new_rows.append({
                            "episode": episode, "component": target["component"],
                            "candidate_context": candidate, "future_utility": utility,
                            "predicted_utility": prediction,
                        })
                    for policy in policies:
                        choices = candidate_sets[policy]
                        if choices:
                            winner = max(choices, key=lambda item: evaluated[item]["prediction"])
                            take = evaluated[winner]["prediction"] > threshold
                            values[policy][episode] = evaluated[winner]["utility"] if take else 0.0
                            accepted[policy][episode] = bool(take)
                        else:
                            values[policy][episode] = 0.0
                            accepted[policy][episode] = False
                    for cap in capacities:
                        random_pools[cap][episode] = [
                            evaluated[candidate] for candidate in banks[f"reservoir_{cap}"]
                        ]
                memory[key] = state
                banks["unbounded"].append(key)
                seen += 1
                for cap in capacities:
                    fifo = banks[f"fifo_{cap}"]
                    fifo.append(key)
                    banks[f"fifo_{cap}"] = fifo[-cap:]
                    reservoir = banks[f"reservoir_{cap}"]
                    if len(reservoir) < cap:
                        reservoir.append(key)
                    else:
                        replacement = generators[cap].randrange(seen)
                        if replacement < cap:
                            reservoir[replacement] = key
            print(json.dumps({
                "capacity_sweep_location": location, "events": len(events),
                "evaluated_rows": len(new_rows), "reused": reused,
            }), flush=True)

    unbounded_metrics = _summarize(values["unbounded"], accepted["unbounded"], group_by_episode, epsilon)
    repetitions = int(config["capacity_selection"]["random_address_repetitions"])
    ordered = sorted(group_by_episode)
    capacity_results = {}
    selected_capacity = None
    for cap in capacities:
        matrix = np.zeros((repetitions, len(ordered)), dtype=np.float64)
        accept_matrix = np.zeros_like(matrix, dtype=bool)
        for repetition in range(repetitions):
            generator = np.random.default_rng(int(config["seed"]) + cap * 1009 + repetition)
            for episode_index, episode in enumerate(ordered):
                pool = random_pools[cap][episode]
                if not pool:
                    continue
                indices = generator.choice(len(pool), size=min(candidate_count, len(pool)), replace=False)
                chosen = [pool[int(index)] for index in indices]
                winner = max(chosen, key=lambda row: row["prediction"])
                take = winner["prediction"] > threshold
                accept_matrix[repetition, episode_index] = take
                matrix[repetition, episode_index] = winner["utility"] if take else 0.0
        random_expected = {
            episode: float(matrix[:, index].mean()) for index, episode in enumerate(ordered)
        }
        reservoir_name, fifo_name = f"reservoir_{cap}", f"fifo_{cap}"
        reservoir_metrics = _summarize(
            values[reservoir_name], accepted[reservoir_name], group_by_episode, epsilon,
        )
        fifo_metrics = _summarize(values[fifo_name], accepted[fifo_name], group_by_episode, epsilon)
        random_summary = {
            "mean_selected_utility": float(matrix.mean()),
            "median_harmful_rate": float(np.median((matrix < -epsilon).mean(axis=1))),
            "mean_acceptance": float(accept_matrix.mean()),
        }
        bootstrap = _paired_component_bootstrap(
            values[reservoir_name], random_expected, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + cap,
        )
        retention = reservoir_metrics["mean_selected_utility"] / max(
            unbounded_metrics["mean_selected_utility"], 1e-12,
        )
        checks = {
            "utility_positive": reservoir_metrics["mean_selected_utility"]
            > float(config["success"]["minimum_routed_utility"]),
            "retains_unbounded": retention
            >= float(config["success"]["minimum_retention_of_unbounded_utility"]),
            "harm_not_above_unbounded": reservoir_metrics["harmful_rate"]
            <= unbounded_metrics["harmful_rate"],
            "beats_fifo": reservoir_metrics["mean_selected_utility"]
            > fifo_metrics["mean_selected_utility"],
            "beats_random_address": reservoir_metrics["mean_selected_utility"]
            > random_summary["mean_selected_utility"],
            "random_interval_positive": bootstrap["ci95"][0] > 0.0,
        }
        capacity_results[str(cap)] = {
            "reservoir": reservoir_metrics, "fifo": fifo_metrics,
            "random_address": random_summary,
            "retention_of_unbounded": retention,
            "reservoir_minus_random_bootstrap": bootstrap,
            "checks": checks, "passed": all(checks.values()),
        }
        if selected_capacity is None and all(checks.values()):
            selected_capacity = cap

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage14_capacity_candidates",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "validation_accessed": True, "test_accessed": False,
        "query_or_future_router_input": False, "rows": new_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage14_validation_capacity_selection",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "validation_accessed": True, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "artifact_sha256": locked["artifact_sha256"],
        "targets": len(targets), "components": len(set(group_by_episode.values())),
        "unbounded_router": unbounded_metrics, "capacity_results": capacity_results,
        "selection_rule": "smallest_passing_capacity",
        "selected_capacity": selected_capacity,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"passed": selected_capacity is not None},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "unbounded": unbounded_metrics,
        "capacities": capacity_results, "selected_capacity": selected_capacity,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
