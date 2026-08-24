#!/usr/bin/env python3
"""Terminal one-shot evaluation of the locked EXP-009 final test model."""

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

from revisit3d.experiments import CachedAtomSegment, adapt_context, geometry_objective, observable_router_features, query_readout_loss
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.evaluate_exp006_validation import _summarize
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _cpu_atom, _device_atom, _float_stats, _identifier, _paired_component_bootstrap,
    _tensor_stats, _timestamp,
)
from revisit3d.scripts.evaluate_exp009_locked_validation import _mips_score, _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_final_test_v24.yaml")
    parser.add_argument("--confirm-terminal-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_terminal_test:
        raise SystemExit("refusing terminal test without explicit confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 final test requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 final test output already exists")
    lock = json.loads(Path(config["output"]["lock_result"]).read_text())
    artifact_path = Path(config["output"]["artifact"])
    artifact = joblib.load(artifact_path)
    if not (
        lock.get("protocol_revision") == config["protocol_revision"]
        and lock.get("test_accessed") is False
        and lock.get("artifact_sha256") == _sha256(artifact_path)
        and artifact.get("split") == "train"
        and artifact.get("test_accessed") is False
        and artifact.get("query_or_future_router_input") is False
        and lock.get("selected_capacity") == artifact.get("bank_capacity")
        == int(config["bank"]["capacity"]) == 64
        and artifact.get("protocol_revision") == config["protocol_revision"]
        and artifact.get("bank_retention") == "deterministic_reservoir"
        and artifact.get("candidate_count") == int(config["bank"]["candidate_count"])
    ):
        raise RuntimeError("final artifact changed after the test lock")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 117
        and all(row.get("split") == "test" for row in manifest)
        and geometry.get("split") == "test"
        and geometry.get("protocol_revision") == config["protocol_revision"]
        and geometry.get("pca_fit_split") == "train"
        and Path(geometry.get("pca_source_cache", "")) == Path(config["stage1"]["pca_source_cache"])
    ):
        raise RuntimeError("final test cache/manifest contract failed")

    context_info, targets = {}, {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context")):
            segment = row[tag]
            key = _identifier(segment)
            context_info.setdefault(key, {
                "id": key, "segment": segment, "cache_index": index,
                "cache_tag": cache_tag, "location": row["location"],
            })
        key = _identifier(row["a_prime"])
        target = {
            "id": key, "cache_index": index, "episode": f"target-{key}",
            "component": f"component-{int(row['component_id'])}", "location": row["location"],
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        }
        if key in targets:
            previous = targets[key]
            if not (previous["component"] == target["component"] and previous["query_frames"] == target["query_frames"]):
                raise RuntimeError("duplicate final-test target is inconsistent")
        else:
            targets[key] = target
    if len(context_info) != 256 or len(targets) != 104:
        raise RuntimeError("final test unique context/target count changed")
    group_by_episode = {target["episode"]: target["component"] for target in targets.values()}
    if len(set(group_by_episode.values())) != 22:
        raise RuntimeError("final test lost component coverage")
    metadata_cache = {}
    scene_root = Path(config["data"]["scene_root"])
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)

    checkpoint = torch.load(config["stage1"]["source_checkpoint"], map_location="cpu", weights_only=False)
    if not (
        checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("final test atom checkpoint changed")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    capacity = int(config["bank"]["capacity"])
    k = int(config["bank"]["candidate_count"])
    strength = float(config["stage1"]["reuse_strength"])
    epsilon = float(config["stage1"]["utility_deadband_minimum"])
    router, columns = artifact["router_model"], artifact["router_feature_columns"]
    threshold, compiled = float(artifact["router_threshold"]), artifact["utility_mips"]
    policies = ("reservoir_capacity64", "fifo_capacity64", "unbounded_utility_address")
    oracle_values = {name: {} for name in policies}
    oracle_accept = {name: {} for name in policies}
    router_values = {name: {} for name in policies}
    router_accept = {name: {} for name in policies}
    current_ratio, random_pools, candidate_rows = {}, {}, []

    with torch.enable_grad():
        for location in sorted({row["location"] for row in context_info.values()}):
            events = sorted(
                [row for row in context_info.values() if row["location"] == location],
                key=lambda row: (row["timestamp"], row["id"]),
            )
            memory = {}
            banks = {name: [] for name in policies}
            seen = 0
            generator = random.Random(
                int(config["seed"]) + int(hashlib.sha1(location.encode()).hexdigest()[:8], 16)
            )
            target_count = 0
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code, step_size=float(config["stage1"]["ttt_step_size"]),
                    steps=int(config["stage1"]["ttt_steps"]),
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
                    target, target_count = targets[key], target_count + 1
                    episode = target["episode"]
                    candidates = {}
                    for name in policies:
                        ranked = sorted((
                            (candidate, _mips_score(compiled, state["descriptor"], memory[candidate]["descriptor"]))
                            for candidate in banks[name]
                        ), key=lambda row: (-row[1], row[0]))[:k]
                        candidates[name] = [row[0] for row in ranked]
                    pool = list(banks["reservoir_capacity64"])
                    union = sorted({candidate for subset in candidates.values() for candidate in subset} | set(pool))
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    base_query = query_readout_loss(head, zero, query)
                    current_query = query_readout_loss(head, replace(zero, code=code), query)
                    current_ratio[episode] = float((current_query / base_query.detach().abs().clamp_min(1e-6)).detach())
                    evaluated = {}
                    for candidate in union:
                        source_state = memory[candidate]
                        source_atom = _device_atom(source_state["atom"], device)
                        alignment = align_atoms(source_atom.detach(), zero.detach())[0]
                        visual = visual_transport(source_atom, zero)
                        candidate_code = (code + strength * visual.code).clamp(-1, 1)
                        candidate_objective = geometry_objective(head, segment, candidate_code)
                        candidate_query = query_readout_loss(head, replace(zero, code=candidate_code), query)
                        features = observable_router_features(
                            current_descriptor=zero.key.mean(dim=(1, 2))[0],
                            source_descriptor=source_atom.key.mean(dim=(1, 2))[0],
                            current_code=code, transported_code=visual.code, visual_result=visual,
                            alignment=alignment, current_pre_objective=pre, current_post_objective=post,
                            candidate_objective=candidate_objective,
                            source_pre_objective=torch.tensor(source_state["pre"], device=device),
                            source_post_objective=torch.tensor(source_state["post"], device=device),
                            current_pre_stats=pre_stats, current_post_stats=post_stats,
                            source_pre_stats=_tensor_stats(source_state["pre_stats"], device),
                            source_post_stats=_tensor_stats(source_state["post_stats"], device),
                        )
                        utility = float(normalized_future_utility(current_query, candidate_query).detach())
                        prediction = float(router.predict(
                            np.asarray(features.detach().cpu(), dtype=np.float64)[None, columns]
                        )[0])
                        evaluated[candidate] = {"utility": utility, "prediction": prediction}
                        candidate_rows.append({
                            "episode": episode, "component": target["component"],
                            "candidate_context": candidate, "future_utility": utility,
                            "predicted_utility": prediction,
                        })
                    random_pools[episode] = [evaluated[candidate] for candidate in pool]
                    for name in policies:
                        subset = candidates[name]
                        if subset:
                            oracle = max(0.0, max(evaluated[item]["utility"] for item in subset))
                            winner = max(subset, key=lambda item: evaluated[item]["prediction"])
                            take = evaluated[winner]["prediction"] > threshold
                            routed = evaluated[winner]["utility"] if take else 0.0
                        else:
                            oracle = routed = 0.0
                            take = False
                        oracle_values[name][episode], oracle_accept[name][episode] = float(oracle), oracle > 0.0
                        router_values[name][episode], router_accept[name][episode] = float(routed), bool(take)
                memory[key] = state
                banks["unbounded_utility_address"].append(key)
                banks["fifo_capacity64"].append(key)
                banks["fifo_capacity64"] = banks["fifo_capacity64"][-capacity:]
                seen += 1
                reservoir = banks["reservoir_capacity64"]
                if len(reservoir) < capacity:
                    reservoir.append(key)
                else:
                    replacement = generator.randrange(seen)
                    if replacement < capacity:
                        reservoir[replacement] = key
            print(json.dumps({"test_location": location, "events": len(events), "targets": target_count, "candidate_rows": len(candidate_rows)}), flush=True)

    repetitions = int(config["bank"]["random_address_repetitions"])
    ordered = sorted(group_by_episode)
    random_matrix = np.zeros((repetitions, len(ordered)), dtype=np.float64)
    random_accept = np.zeros_like(random_matrix, dtype=bool)
    for repetition in range(repetitions):
        generator = np.random.default_rng(int(config["seed"]) + repetition)
        for index, episode in enumerate(ordered):
            pool = random_pools[episode]
            if not pool:
                continue
            chosen = [pool[int(i)] for i in generator.choice(len(pool), size=min(k, len(pool)), replace=False)]
            winner = max(chosen, key=lambda row: row["prediction"])
            take = winner["prediction"] > threshold
            random_accept[repetition, index] = take
            random_matrix[repetition, index] = winner["utility"] if take else 0.0
    random_expected = {episode: float(random_matrix[:, i].mean()) for i, episode in enumerate(ordered)}
    metrics = {
        name: {
            "oracle_topk": _summarize(oracle_values[name], oracle_accept[name], group_by_episode, epsilon),
            "router": _summarize(router_values[name], router_accept[name], group_by_episode, epsilon),
        } for name in policies
    }
    primary, fifo, unbounded = policies
    random_summary = {
        "mean_selected_utility": float(random_matrix.mean()),
        "median_harmful_rate": float(np.median((random_matrix < -epsilon).mean(axis=1))),
        "mean_acceptance": float(random_accept.mean()),
    }
    bootstrap = {
        "primary_minus_random_address": _paired_component_bootstrap(
            router_values[primary], random_expected, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "primary_minus_fifo": _paired_component_bootstrap(
            router_values[primary], router_values[fifo], group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]), seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    retention = metrics[primary]["router"]["mean_selected_utility"] / max(
        metrics[unbounded]["router"]["mean_selected_utility"], 1e-12,
    )
    mean_current = float(np.mean(list(current_ratio.values())))
    checks = {
        "current_objective_healthy": mean_current <= float(config["success"]["maximum_mean_current_to_base_ratio"]),
        "component_health": len(set(group_by_episode.values())) >= int(config["success"]["minimum_components"]),
        "primary_utility_positive": metrics[primary]["router"]["mean_selected_utility"] > float(config["success"]["minimum_primary_routed_utility"]),
        "retains_unbounded": retention >= float(config["success"]["minimum_retention_of_unbounded_utility"]),
        "harm_not_above_unbounded": metrics[primary]["router"]["harmful_rate"] <= metrics[unbounded]["router"]["harmful_rate"],
        "beats_fifo": metrics[primary]["router"]["mean_selected_utility"] > metrics[fifo]["router"]["mean_selected_utility"],
        "beats_random_address": metrics[primary]["router"]["mean_selected_utility"] > random_summary["mean_selected_utility"],
        "random_interval_positive": bootstrap["primary_minus_random_address"]["ci95"][0] > 0.0,
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage16_test_candidates", "split": "test",
        "protocol_revision": config["protocol_revision"], "validation_accessed": True,
        "test_accessed": True, "query_or_future_router_input": False, "rows": candidate_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage16_final_locked_test",
        "protocol_revision": config["protocol_revision"], "split": "test",
        "validation_accessed": True, "test_accessed": True,
        "query_or_future_router_input": False, "config": str(config_path),
        "artifact_sha256": lock["artifact_sha256"], "manifest_episodes": len(manifest),
        "unique_contexts": len(context_info), "unique_targets": len(targets),
        "components": len(set(group_by_episode.values())), "mean_current_to_base_ratio": mean_current,
        "capacity": capacity, "candidate_count": k, "metrics": metrics,
        "random_address": random_summary, "retention_of_unbounded_router_utility": retention,
        "bootstrap": bootstrap, "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "terminal_no_further_test_tuning": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "mean_current_to_base": mean_current,
        "metrics": metrics, "random_address": random_summary, "retention": retention,
        "bootstrap": bootstrap, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
