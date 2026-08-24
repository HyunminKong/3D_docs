#!/usr/bin/env python3
"""Source-safe true-time capacity-bank replay for EXP-009 Stage 11."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    _router_models,
    _tensor_stats,
    _timestamp,
)


def _descriptor_pair(current: torch.Tensor, source: torch.Tensor) -> np.ndarray:
    current_np = current.float().numpy()
    source_np = source.float().numpy()
    return np.concatenate((
        current_np, source_np, current_np - source_np, current_np * source_np,
    )).astype(np.float64)


def _source_history(state: dict) -> np.ndarray:
    denominator = max(abs(state["pre"]), 1e-6)
    return np.asarray((
        state["post"] / denominator,
        (state["pre"] - state["post"]) / denominator,
        state["pre_stats"]["track_coverage"],
        state["pre_stats"]["mean_3d_residual"],
        state["post_stats"]["mean_3d_residual"],
    ), dtype=np.float64)


def _history_source_weight(model) -> np.ndarray:
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    effective = np.asarray(ridge.coef_, dtype=np.float64) / np.asarray(scaler.scale_)
    # Stage-8 history order: current post/improvement, source post/improvement,
    # current coverage/pre/post residual, source coverage/pre/post residual.
    return effective[[2, 3, 7, 8, 9]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_capacity_bank_v21.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 capacity replay requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 Stage-11 output already exists")
    stage7_config = yaml.safe_load(Path(config["source"]["stage7_config"]).read_text())
    stage10 = json.loads(Path(config["source"]["stage10_result"]).read_text())
    training = json.loads(Path(config["source"]["stage8_candidate_cache"]).read_text())
    if not (
        stage10.get("registered_gate", {}).get("passed") is True
        and stage10.get("split") == training.get("split") == "train"
        and stage10.get("validation_accessed") is False
        and training.get("validation_accessed") is False
        and stage10.get("test_accessed") is False
        and training.get("test_accessed") is False
        and stage10.get("query_or_future_router_input") is False
        and training.get("query_or_future_router_input") is False
    ):
        raise RuntimeError("Stage 11 requires the passing source-safe train artifacts")

    manifest = json.loads(Path(stage7_config["data"]["geometry_manifest"]).read_text())
    geometry = torch.load(
        stage7_config["data"]["geometry_cache"], map_location="cpu",
        weights_only=False, mmap=True,
    )
    context_info, targets = {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (
            ("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context"),
        ):
            segment = row[tag]
            key = _identifier(segment)
            context_info.setdefault(key, {
                "id": key, "segment": segment, "cache_index": index, "cache_tag": cache_tag,
                "scene": segment["scene"], "location": row["location"],
            })
        target_key = _identifier(row["a_prime"])
        target = {
            "id": target_key, "cache_index": index, "episode": f"target-{target_key}",
            "component": f"component-{int(row['component_id'])}",
            "location": row["location"],
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        }
        if target_key in targets and targets[target_key]["query_frames"] != target["query_frames"]:
            raise RuntimeError("duplicate target query changed")
        targets.setdefault(target_key, target)
    if len(context_info) != 557 or len(targets) != 218:
        raise RuntimeError("Stage-11 context/target contract changed")
    metadata_cache = {}
    scene_root = Path(stage7_config["data"]["scene_root"])
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)

    train_rows = training["rows"]
    train_matrix = np.asarray([row["prefilter_features"] for row in train_rows], dtype=np.float64)
    train_utility = np.asarray([row["future_utility"] for row in train_rows], dtype=np.float64)
    target_location = np.asarray([context_info[row["target_context"]]["location"] for row in train_rows])
    source_location = np.asarray([context_info[row["candidate_context"]]["location"] for row in train_rows])
    location_models = {}
    for location in sorted(set(target_location.tolist())):
        keep = (target_location != location) & (source_location != location)
        address = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        history = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        address.fit(train_matrix[keep, 8:264], train_utility[keep])
        history.fit(train_matrix[keep, 264:274], train_utility[keep])
        location_models[location] = {
            "address": address, "history_weight": _history_source_weight(history),
            "train_pairs": int(keep.sum()),
        }

    checkpoint = torch.load(
        stage7_config["models"]["plasticity_atom_checkpoint"],
        map_location="cpu", weights_only=False,
    )
    if not (
        checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage 11 requires the locked atom")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(stage7_config["models"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    router_models, router_thresholds, router_columns = _router_models(stage7_config)

    policies = tuple(config["bank"]["policies"])
    capacity = int(config["bank"]["capacity"])
    candidate_count = int(config["bank"]["candidate_count"])
    strength = float(stage7_config["adaptation"]["reuse_strength"])
    epsilon = float(config["statistics"]["utility_deadband"])
    oracle_values = {policy: {} for policy in policies}
    oracle_accept = {policy: {} for policy in policies}
    router_values = {policy: {} for policy in policies}
    router_accept = {policy: {} for policy in policies}
    group_by_episode = {target["episode"]: target["component"] for target in targets.values()}
    candidate_rows, selection_rows = [], []

    with torch.enable_grad():
        for location in sorted(location_models):
            events = sorted(
                [row for row in context_info.values() if row["location"] == location],
                key=lambda row: (row["timestamp"], row["id"]),
            )
            memory = {}
            banks = {policy: [] for policy in policies}
            seen = 0
            generator = random.Random(
                int(config["seed"]) + int(hashlib.sha1(location.encode()).hexdigest()[:8], 16)
            )
            address_model = location_models[location]["address"]
            history_weight = location_models[location]["history_weight"]
            location_targets = 0
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code,
                    step_size=float(stage7_config["adaptation"]["ttt_step_size"]),
                    steps=int(stage7_config["adaptation"]["ttt_steps"]),
                )
                pre, pre_stats = geometry_objective(head, segment, zero.code, return_stats=True)
                post, post_stats = geometry_objective(head, segment, code, return_stats=True)
                state = {
                    "atom": _cpu_atom(replace(zero, code=code.detach())),
                    "descriptor": zero.key.mean(dim=(1, 2))[0].detach().cpu(),
                    "pre": float(pre.detach()), "post": float(post.detach()),
                    "pre_stats": _float_stats(pre_stats), "post_stats": _float_stats(post_stats),
                }
                state["retention_priority"] = float(history_weight @ _source_history(state))

                if key in targets:
                    target = targets[key]
                    episode = target["episode"]
                    location_targets += 1
                    policy_candidates, policy_scores = {}, {}
                    for policy in policies:
                        bank = banks[policy]
                        if bank:
                            features = np.asarray([
                                _descriptor_pair(state["descriptor"], memory[candidate]["descriptor"])
                                for candidate in bank
                            ])
                            scores = address_model.predict(features)
                            ranked = sorted(
                                zip(bank, scores.tolist()), key=lambda row: (-row[1], row[0]),
                            )[:candidate_count]
                            policy_candidates[policy] = [row[0] for row in ranked]
                            policy_scores[policy] = dict(ranked)
                        else:
                            policy_candidates[policy] = []
                            policy_scores[policy] = {}
                    union = sorted({candidate for subset in policy_candidates.values() for candidate in subset})
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    current_query = query_readout_loss(head, replace(zero, code=code), query)
                    evaluated = {}
                    for candidate in union:
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
                        prediction = float(router_models[target["component"]].predict(
                            np.asarray(features.detach().cpu(), dtype=np.float64)[None, router_columns]
                        )[0])
                        evaluated[candidate] = {"utility": utility, "prediction": prediction}
                        candidate_rows.append({
                            "episode": episode, "component": target["component"],
                            "candidate_context": candidate, "future_utility": utility,
                            "predicted_utility": prediction,
                        })
                    selection = {
                        "episode": episode, "component": target["component"],
                        "location": location, "policies": {},
                    }
                    for policy in policies:
                        candidates = policy_candidates[policy]
                        if candidates:
                            oracle_choice = max(candidates, key=lambda item: evaluated[item]["utility"])
                            oracle_utility = max(0.0, evaluated[oracle_choice]["utility"])
                            router_choice = max(candidates, key=lambda item: evaluated[item]["prediction"])
                            take = evaluated[router_choice]["prediction"] > router_thresholds[target["component"]]
                            router_utility = evaluated[router_choice]["utility"] if take else 0.0
                        else:
                            oracle_choice = router_choice = None
                            oracle_utility = router_utility = 0.0
                            take = False
                        oracle_values[policy][episode] = float(oracle_utility)
                        oracle_accept[policy][episode] = oracle_utility > 0.0
                        router_values[policy][episode] = float(router_utility)
                        router_accept[policy][episode] = bool(take)
                        selection["policies"][policy] = {
                            "bank_size": len(banks[policy]), "candidates": candidates,
                            "address_scores": policy_scores[policy],
                            "oracle_candidate": oracle_choice, "oracle_utility": float(oracle_utility),
                            "router_candidate": router_choice, "router_accepted": bool(take),
                            "router_utility": float(router_utility),
                        }
                    selection_rows.append(selection)

                memory[key] = state
                banks["unbounded_utility_address"].append(key)
                banks["history_capacity8"].append(key)
                banks["history_capacity8"] = sorted(
                    banks["history_capacity8"],
                    key=lambda item: (-memory[item]["retention_priority"], item),
                )[:capacity]
                banks["fifo_capacity8"].append(key)
                banks["fifo_capacity8"] = banks["fifo_capacity8"][-capacity:]
                seen += 1
                reservoir = banks["reservoir_capacity8"]
                if len(reservoir) < capacity:
                    reservoir.append(key)
                else:
                    replacement = generator.randrange(seen)
                    if replacement < capacity:
                        reservoir[replacement] = key
            print(json.dumps({
                "location": location, "events": len(events), "targets": location_targets,
                "candidate_rows": len(candidate_rows),
            }), flush=True)

    expected = set(group_by_episode)
    if any(set(values) != expected for values in router_values.values()):
        raise RuntimeError("capacity replay did not cover every target")
    metrics = {
        policy: {
            "oracle_topk": _summarize(
                oracle_values[policy], oracle_accept[policy], group_by_episode, epsilon,
            ),
            "router": _summarize(
                router_values[policy], router_accept[policy], group_by_episode, epsilon,
            ),
        } for policy in policies
    }
    history = "history_capacity8"
    fifo = "fifo_capacity8"
    reservoir = "reservoir_capacity8"
    unbounded = "unbounded_utility_address"
    bootstrap = {
        "history_minus_fifo_router": _paired_component_bootstrap(
            router_values[history], router_values[fifo], group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "history_minus_reservoir_router": _paired_component_bootstrap(
            router_values[history], router_values[reservoir], group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    retention = (
        metrics[history]["router"]["mean_selected_utility"]
        / max(metrics[unbounded]["router"]["mean_selected_utility"], 1e-12)
    )
    checks = {
        "retains_unbounded_utility": retention
        >= float(config["success"]["minimum_retention_of_unbounded_router_utility"]),
        "history_beats_fifo": metrics[history]["router"]["mean_selected_utility"]
        > metrics[fifo]["router"]["mean_selected_utility"],
        "history_beats_reservoir": metrics[history]["router"]["mean_selected_utility"]
        > metrics[reservoir]["router"]["mean_selected_utility"],
        "history_harm_not_above_fifo": metrics[history]["router"]["harmful_rate"]
        <= metrics[fifo]["router"]["harmful_rate"],
        "history_fifo_interval_positive": bootstrap["history_minus_fifo_router"]["ci95"][0] > 0.0,
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage11_capacity_candidates",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "rows": candidate_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage11_source_safe_capacity_bank",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "targets": len(targets), "contexts": len(context_info),
        "components": len(set(group_by_episode.values())), "capacity": capacity,
        "candidate_count": candidate_count, "stream_partition": "official_location",
        "location_train_pairs": {
            location: payload["train_pairs"] for location, payload in location_models.items()
        },
        "metrics": metrics, "retention_of_unbounded_router_utility": retention,
        "bootstrap": bootstrap, "selection_rows": selection_rows,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "metrics": metrics, "retention": retention,
        "bootstrap": bootstrap, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
