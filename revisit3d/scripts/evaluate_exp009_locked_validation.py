#!/usr/bin/env python3
"""One-shot locked validation of the EXP-009 utility-MIPS reservoir bank."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mips_score(compiled: dict, current: torch.Tensor, source: torch.Tensor) -> float:
    current_np = current.float().numpy().astype(np.float64)
    source_np = source.float().numpy().astype(np.float64)
    return float(
        compiled["intercept"]
        + current_np @ compiled["current_weight"]
        + source_np @ (compiled["source_weight"] + current_np * compiled["product_weight"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_validation_lock_v22.yaml")
    parser.add_argument("--confirm-one-shot-validation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_one_shot_validation:
        raise SystemExit("refusing validation evaluation without explicit one-shot confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 locked validation requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["validation_result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 locked validation output already exists")
    lock = json.loads(Path(config["output"]["train_lock_result"]).read_text())
    artifact_path = Path(config["output"]["artifact"])
    artifact = joblib.load(artifact_path)
    if not (
        lock.get("split") == artifact.get("split") == "train"
        and lock.get("validation_accessed") is False
        and artifact.get("validation_accessed") is False
        and lock.get("test_accessed") is False
        and artifact.get("test_accessed") is False
        and lock.get("artifact_sha256") == _sha256(artifact_path)
        and artifact.get("protocol_revision") == config["protocol_revision"]
        and artifact.get("bank_retention") == "deterministic_reservoir"
        and artifact.get("bank_capacity") == int(config["bank"]["capacity"])
        and artifact.get("candidate_count") == int(config["bank"]["candidate_count"])
    ):
        raise RuntimeError("deployable artifact changed after the validation lock")

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False, mmap=True)
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 117
        and all(row.get("split") == "val" for row in manifest)
        and geometry.get("split") == "val"
        and geometry.get("protocol_revision") == config["protocol_revision"]
        and geometry.get("pca_fit_split") == "train"
        and Path(geometry.get("pca_source_cache", "")) == Path(config["stage1"]["pca_source_cache"])
    ):
        raise RuntimeError("locked validation cache/manifest contract failed")

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
        if target_key in targets:
            previous = targets[target_key]
            if not (
                previous["component"] == target["component"]
                and previous["location"] == target["location"]
                and previous["query_frames"] == target["query_frames"]
            ):
                raise RuntimeError("duplicate validation target is inconsistent")
        else:
            targets[target_key] = target
    if len(context_info) != 241 or len(targets) != 103:
        raise RuntimeError("validation unique context/target count changed")
    group_by_episode = {target["episode"]: target["component"] for target in targets.values()}
    if len(set(group_by_episode.values())) != 17:
        raise RuntimeError("validation lost physical-component coverage")
    metadata_cache = {}
    scene_root = Path(config["data"]["scene_root"])
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)

    checkpoint = torch.load(
        config["stage1"]["source_checkpoint"], map_location="cpu", weights_only=False,
    )
    if not (
        checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("validation atom is not the locked train checkpoint")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)

    policies = ("reservoir_capacity8", "fifo_capacity8", "unbounded_utility_address")
    capacity = int(config["bank"]["capacity"])
    candidate_count = int(config["bank"]["candidate_count"])
    strength = float(config["stage1"]["reuse_strength"])
    epsilon = float(config["stage1"]["utility_deadband_minimum"])
    router = artifact["router_model"]
    router_columns = artifact["router_feature_columns"]
    router_threshold = float(artifact["router_threshold"])
    compiled = artifact["utility_mips"]
    oracle_values = {policy: {} for policy in policies}
    oracle_accept = {policy: {} for policy in policies}
    router_values = {policy: {} for policy in policies}
    router_accept = {policy: {} for policy in policies}
    current_to_base = {}
    random_pools = {}
    candidate_rows, selection_rows = [], []

    with torch.enable_grad():
        for location in sorted({row["location"] for row in context_info.values()}):
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
            location_targets = 0
            for event in events:
                key = event["id"]
                payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
                role = "current" if key in targets else "source"
                segment = CachedAtomSegment.from_cache(payload, role, device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code,
                    step_size=float(config["stage1"]["ttt_step_size"]),
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
                    target = targets[key]
                    episode = target["episode"]
                    location_targets += 1
                    policy_candidates, policy_scores = {}, {}
                    for policy in policies:
                        bank = banks[policy]
                        ranked = sorted(
                            ((candidate, _mips_score(
                                compiled, state["descriptor"], memory[candidate]["descriptor"],
                            )) for candidate in bank),
                            key=lambda row: (-row[1], row[0]),
                        )[:candidate_count]
                        policy_candidates[policy] = [row[0] for row in ranked]
                        policy_scores[policy] = dict(ranked)
                    reservoir_pool = list(banks["reservoir_capacity8"])
                    union = sorted(
                        {candidate for subset in policy_candidates.values() for candidate in subset}
                        | set(reservoir_pool)
                    )
                    query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                    query = CachedAtomSegment.from_cache(query_payload, "query", device)
                    base_query = query_readout_loss(head, zero, query)
                    current_query = query_readout_loss(head, replace(zero, code=code), query)
                    current_to_base[episode] = float(
                        (current_query / base_query.detach().abs().clamp_min(1e-6)).detach()
                    )
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
                        prediction = float(router.predict(
                            np.asarray(features.detach().cpu(), dtype=np.float64)[None, router_columns]
                        )[0])
                        evaluated[candidate] = {"utility": utility, "prediction": prediction}
                        candidate_rows.append({
                            "episode": episode, "component": target["component"],
                            "candidate_context": candidate, "future_utility": utility,
                            "predicted_utility": prediction,
                        })
                    random_pools[episode] = [
                        {"candidate": candidate, **evaluated[candidate]} for candidate in reservoir_pool
                    ]
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
                            take = evaluated[router_choice]["prediction"] > router_threshold
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
                "validation_location": location, "events": len(events),
                "targets": location_targets, "candidate_rows": len(candidate_rows),
            }), flush=True)

    expected = set(group_by_episode)
    if any(set(values) != expected for values in router_values.values()):
        raise RuntimeError("validation policies did not cover every unique target")
    repetitions = int(config["bank"]["random_address_repetitions"])
    ordered = sorted(expected)
    random_router = np.zeros((repetitions, len(ordered)), dtype=np.float64)
    random_accept = np.zeros_like(random_router, dtype=bool)
    for repetition in range(repetitions):
        generator = np.random.default_rng(int(config["seed"]) + repetition)
        for episode_index, episode in enumerate(ordered):
            pool = random_pools[episode]
            if not pool:
                continue
            indices = generator.choice(
                len(pool), size=min(candidate_count, len(pool)), replace=False,
            )
            chosen = [pool[int(index)] for index in indices]
            winner = max(chosen, key=lambda row: row["prediction"])
            take = winner["prediction"] > router_threshold
            random_accept[repetition, episode_index] = take
            random_router[repetition, episode_index] = winner["utility"] if take else 0.0
    random_expected = {
        episode: float(random_router[:, index].mean()) for index, episode in enumerate(ordered)
    }
    random_harm = (random_router < -epsilon).mean(axis=1)
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
    primary = "reservoir_capacity8"
    fifo = "fifo_capacity8"
    unbounded = "unbounded_utility_address"
    random_summary = {
        "mean_selected_utility": float(random_router.mean()),
        "median_harmful_rate": float(np.median(random_harm)),
        "mean_acceptance": float(random_accept.mean()),
    }
    bootstrap = {
        "primary_minus_random_address": _paired_component_bootstrap(
            router_values[primary], random_expected, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "primary_minus_fifo": _paired_component_bootstrap(
            router_values[primary], router_values[fifo], group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    retention = (
        metrics[primary]["router"]["mean_selected_utility"]
        / max(metrics[unbounded]["router"]["mean_selected_utility"], 1e-12)
    )
    mean_current_to_base = float(np.mean(list(current_to_base.values())))
    checks = {
        "current_objective_healthy": mean_current_to_base
        <= float(config["success"]["maximum_mean_current_to_base_ratio"]),
        "component_health": len(set(group_by_episode.values()))
        >= int(config["success"]["minimum_components"]),
        "primary_utility_positive": metrics[primary]["router"]["mean_selected_utility"]
        > float(config["success"]["minimum_primary_routed_utility"]),
        "retains_unbounded_utility": retention
        >= float(config["success"]["minimum_retention_of_unbounded_utility"]),
        "primary_beats_fifo": metrics[primary]["router"]["mean_selected_utility"]
        > metrics[fifo]["router"]["mean_selected_utility"],
        "primary_harm_not_above_fifo": metrics[primary]["router"]["harmful_rate"]
        <= metrics[fifo]["router"]["harmful_rate"],
        "primary_beats_random_address": metrics[primary]["router"]["mean_selected_utility"]
        > random_summary["mean_selected_utility"],
        "primary_random_interval_positive":
            bootstrap["primary_minus_random_address"]["ci95"][0] > 0.0,
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage13_validation_candidates",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "validation_accessed": True, "test_accessed": False,
        "query_or_future_router_input": False, "rows": candidate_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage13_locked_one_shot_validation",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "validation_accessed": True, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "artifact_sha256": lock["artifact_sha256"],
        "manifest_episodes": len(manifest), "unique_contexts": len(context_info),
        "unique_targets": len(targets), "components": len(set(group_by_episode.values())),
        "locations": sorted({target["location"] for target in targets.values()}),
        "mean_current_to_base_ratio": mean_current_to_base,
        "capacity": capacity, "candidate_count": candidate_count,
        "metrics": metrics, "random_address": random_summary,
        "retention_of_unbounded_router_utility": retention,
        "bootstrap": bootstrap, "selection_rows": selection_rows,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "mean_current_to_base": mean_current_to_base,
        "metrics": metrics, "random_address": random_summary,
        "retention": retention, "bootstrap": bootstrap,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
