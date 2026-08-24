#!/usr/bin/env python3
"""Matched random top-K null for the EXP-009 causal DINOv2 retrieval result."""

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

from revisit3d.experiments import (
    CachedAtomSegment,
    adapt_context,
    geometry_objective,
    observable_router_features,
    query_readout_loss,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
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
from revisit3d.scripts.simulate_exp007_token_bucket import _token_pair_features


def _prefilter_features(
    current_state: dict,
    source_state: dict,
    current_dino: np.ndarray,
    source_dino: np.ndarray,
) -> list[float]:
    current = current_state["descriptor"].float().numpy()
    source = source_state["descriptor"].float().numpy()
    descriptor = np.concatenate((current, source, current - source, current * source))
    current_denominator = max(abs(current_state["pre"]), 1e-6)
    source_denominator = max(abs(source_state["pre"]), 1e-6)
    scalars = np.asarray((
        current_state["post"] / current_denominator,
        (current_state["pre"] - current_state["post"]) / current_denominator,
        source_state["post"] / source_denominator,
        (source_state["pre"] - source_state["post"]) / source_denominator,
        current_state["pre_stats"]["track_coverage"],
        current_state["pre_stats"]["mean_3d_residual"],
        current_state["post_stats"]["mean_3d_residual"],
        source_state["pre_stats"]["track_coverage"],
        source_state["pre_stats"]["mean_3d_residual"],
        source_state["post_stats"]["mean_3d_residual"],
    ), dtype=np.float64)
    values = np.concatenate((
        np.asarray(_token_pair_features(current_dino, source_dino), dtype=np.float64),
        descriptor.astype(np.float64), scalars,
    ))
    if values.shape != (274,) or not np.isfinite(values).all():
        raise RuntimeError("observable prefilter feature contract failed")
    return values.tolist()


def _distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_random_retrieval_null_v18.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 random retrieval null requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    stage7_config = yaml.safe_load(Path(config["source"]["stage7_config"]).read_text())
    result_path = Path(config["output"]["result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 Stage-8 output already exists")
    stage7 = json.loads(Path(config["source"]["stage7_result"]).read_text())
    old_candidates = json.loads(Path(config["source"]["stage7_candidate_cache"]).read_text())
    if not (
        stage7.get("protocol_revision") == "v1.7"
        and stage7.get("split") == old_candidates.get("split") == "train"
        and stage7.get("validation_accessed") is False
        and stage7.get("test_accessed") is False
        and old_candidates.get("validation_accessed") is False
        and old_candidates.get("test_accessed") is False
        and stage7.get("query_or_future_router_input") is False
        and old_candidates.get("query_or_future_router_input") is False
    ):
        raise RuntimeError("Stage 8 requires the locked train-only Stage-7 result")
    old_by_pair = {
        (row["episode"], row["candidate_context"]): row for row in old_candidates["rows"]
    }

    manifest = json.loads(Path(stage7_config["data"]["geometry_manifest"]).read_text())
    geometry = torch.load(
        stage7_config["data"]["geometry_cache"], map_location="cpu",
        weights_only=False, mmap=True,
    )
    dino = torch.load(
        stage7_config["output"]["dinov2_context_cache"], map_location="cpu",
        weights_only=False,
    )
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 225
        and geometry.get("split") == dino.get("split") == "train"
        and dino.get("contexts") == 557
        and dino.get("validation_accessed") is False
        and dino.get("test_accessed") is False
    ):
        raise RuntimeError("Stage-8 source cache contract failed")

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
                and previous["query_frames"] == target["query_frames"]
            ):
                raise RuntimeError("duplicate target contract changed")
        else:
            targets[target_key] = target
    if len(context_info) != 557 or len(targets) != 218:
        raise RuntimeError("Stage-8 unique context/target count changed")
    metadata_cache = {}
    scene_root = Path(stage7_config["data"]["scene_root"])
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)
    events = sorted(context_info.values(), key=lambda row: (row["timestamp"], row["id"]))

    checkpoint = torch.load(
        stage7_config["models"]["plasticity_atom_checkpoint"],
        map_location="cpu", weights_only=False,
    )
    if not (
        checkpoint.get("protocol_revision") == "v2.7"
        and checkpoint.get("split") == "train"
        and checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage 8 requires the locked EXP-006 atom")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(stage7_config["models"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    router_models, router_thresholds, router_columns = _router_models(stage7_config)

    panel_size = int(config["random_null"]["uniform_panel_size"])
    strength = float(stage7_config["adaptation"]["reuse_strength"])
    memory, bank, panel_rows = {}, [], []
    panel_by_episode = {}
    reused, evaluated = 0, 0
    group_by_episode = {
        target["episode"]: target["component"] for target in targets.values()
    }

    with torch.enable_grad():
        for event_index, event in enumerate(events):
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

            if key in targets:
                target = targets[key]
                episode = target["episode"]
                stable_seed = int(hashlib.sha1(episode.encode()).hexdigest()[:8], 16)
                generator = random.Random(int(config["seed"]) + stable_seed)
                panel = generator.sample(bank, min(panel_size, len(bank)))
                query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                query = CachedAtomSegment.from_cache(query_payload, "query", device)
                current_query = query_readout_loss(head, replace(zero, code=code), query)
                current_dino = dino["rows"][key]["dinov2"].float().numpy()
                rows = []
                for candidate in panel:
                    source_state = memory[candidate]
                    prefilter = _prefilter_features(
                        state, source_state, current_dino,
                        dino["rows"][candidate]["dinov2"].float().numpy(),
                    )
                    previous = old_by_pair.get((episode, candidate))
                    if previous is not None:
                        utility = float(previous["future_utility"])
                        prediction = float(previous["predicted_utility"])
                        reused += 1
                    else:
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
                        utility = float(normalized_future_utility(
                            current_query, candidate_query,
                        ).detach())
                        prediction = float(router_models[target["component"]].predict(
                            np.asarray(features.detach().cpu(), dtype=np.float64)[None, router_columns]
                        )[0])
                        evaluated += 1
                    item = {
                        "episode": episode, "component": target["component"],
                        "target_context": key, "candidate_context": candidate,
                        "future_utility": utility, "predicted_utility": prediction,
                        "prefilter_features": prefilter,
                    }
                    panel_rows.append(item)
                    rows.append(item)
                panel_by_episode[episode] = rows

            memory[key] = state
            bank.append(key)
            if (event_index + 1) % 25 == 0 or event_index + 1 == len(events):
                print(json.dumps({
                    "events": event_index + 1, "total": len(events),
                    "targets": len(panel_by_episode), "new_candidates": evaluated,
                    "reused_candidates": reused,
                }), flush=True)

    repetitions = int(config["random_null"]["repetitions"])
    candidate_count = int(config["random_null"]["candidate_count"])
    ordered_episodes = sorted(group_by_episode)
    oracle_matrix = np.zeros((repetitions, len(ordered_episodes)), dtype=np.float64)
    router_matrix = np.zeros_like(oracle_matrix)
    accept_matrix = np.zeros_like(oracle_matrix, dtype=bool)
    for repetition in range(repetitions):
        generator = np.random.default_rng(int(config["seed"]) + repetition)
        for episode_index, episode in enumerate(ordered_episodes):
            rows = panel_by_episode[episode]
            if not rows:
                continue
            indices = generator.choice(
                len(rows), size=min(candidate_count, len(rows)), replace=False,
            )
            chosen = [rows[int(index)] for index in indices]
            oracle_matrix[repetition, episode_index] = max(
                0.0, max(row["future_utility"] for row in chosen)
            )
            winner = max(chosen, key=lambda row: row["predicted_utility"])
            take = winner["predicted_utility"] > router_thresholds[winner["component"]]
            accept_matrix[repetition, episode_index] = take
            router_matrix[repetition, episode_index] = winner["future_utility"] if take else 0.0

    epsilon = float(config["statistics"]["utility_deadband"])
    oracle_mean = oracle_matrix.mean(axis=1)
    router_mean = router_matrix.mean(axis=1)
    router_harm = (router_matrix < -epsilon).mean(axis=1)
    router_accept = accept_matrix.mean(axis=1)
    dino_oracle = {
        row["episode"]: float(row["policies"]["dinov2"]["oracle_utility"])
        for row in stage7["selection_rows"]
    }
    dino_router = {
        row["episode"]: float(row["policies"]["dinov2"]["router_utility"])
        for row in stage7["selection_rows"]
    }
    random_expected_oracle = {
        episode: float(oracle_matrix[:, index].mean())
        for index, episode in enumerate(ordered_episodes)
    }
    random_expected_router = {
        episode: float(router_matrix[:, index].mean())
        for index, episode in enumerate(ordered_episodes)
    }
    observed_oracle = float(stage7["metrics"]["dinov2"]["oracle_topk"]["mean_selected_utility"])
    observed_router = float(stage7["metrics"]["dinov2"]["router"]["mean_selected_utility"])
    observed_harm = float(stage7["metrics"]["dinov2"]["router"]["harmful_rate"])
    oracle_p = float((1 + np.sum(oracle_mean >= observed_oracle)) / (repetitions + 1))
    router_p = float((1 + np.sum(router_mean >= observed_router)) / (repetitions + 1))
    bootstrap = {
        "dinov2_minus_random_expected_oracle_topk": _paired_component_bootstrap(
            dino_oracle, random_expected_oracle, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
        ),
        "dinov2_minus_random_expected_router": _paired_component_bootstrap(
            dino_router, random_expected_router, group_by_episode,
            samples=int(config["statistics"]["bootstrap_samples"]),
            seed=int(config["statistics"]["bootstrap_seed"]) + 1,
        ),
    }
    maximum_p = float(config["success"]["maximum_one_sided_p"])
    checks = {
        "dinov2_oracle_exceeds_95pct_random": oracle_p <= maximum_p,
        "dinov2_router_exceeds_95pct_random": router_p <= maximum_p,
        "oracle_component_interval_positive":
            bootstrap["dinov2_minus_random_expected_oracle_topk"]["ci95"][0] > 0.0,
        "router_component_interval_positive":
            bootstrap["dinov2_minus_random_expected_router"]["ci95"][0] > 0.0,
        "dinov2_router_harm_not_above_random_median":
            observed_harm <= float(np.median(router_harm)),
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage8_random_null_candidates",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False,
        "prefilter_feature_dimensions": 274, "rows": panel_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage8_matched_random_retrieval_null",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "contexts": len(events), "unique_targets": len(targets),
        "components": len(set(group_by_episode.values())),
        "panel_size": panel_size, "candidate_count": candidate_count,
        "candidate_rows": len(panel_rows), "new_candidate_evaluations": evaluated,
        "reused_stage7_candidates": reused, "repetitions": repetitions,
        "dinov2_observed": {
            "oracle_topk_mean_utility": observed_oracle,
            "router_mean_utility": observed_router, "router_harm": observed_harm,
        },
        "random_null": {
            "oracle_topk_mean_utility": _distribution(oracle_mean),
            "router_mean_utility": _distribution(router_mean),
            "router_harm": _distribution(router_harm),
            "router_acceptance": _distribution(router_accept),
        },
        "one_sided_p": {"oracle_topk": oracle_p, "router": router_p},
        "bootstrap": bootstrap,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "dinov2": result["dinov2_observed"],
        "random_null": result["random_null"], "p": result["one_sided_p"],
        "bootstrap": bootstrap, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
