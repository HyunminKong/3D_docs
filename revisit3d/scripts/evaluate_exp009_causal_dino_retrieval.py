#!/usr/bin/env python3
"""Causal top-K adaptation-utility test of the EXP-009 DINOv2 address."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.experiments import (
    PRIMARY_SCALAR_INDICES,
    CachedAtomSegment,
    adapt_context,
    geometry_objective,
    observable_router_features,
    primary_feature_columns,
    query_readout_loss,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import PlasticityAtom, SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.evaluate_exp006_validation import _summarize
from revisit3d.scripts.evaluate_exp009_nested_router import _model
from revisit3d.scripts.simulate_exp007_token_bucket import _token_pair_features


TIMESTAMP = re.compile(r"__(\d+)\.jpg$")


def _identifier(segment: dict) -> str:
    payload = f"{segment['scene']}:{','.join(map(str, segment['frames']))}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _timestamp(segment: dict, scene_root: Path, metadata_cache: dict[str, dict]) -> int:
    scene = segment["scene"]
    if scene not in metadata_cache:
        metadata_cache[scene] = json.loads(
            (scene_root / scene / "opencv_cameras.json").read_text()
        )
    path = metadata_cache[scene]["frames"][int(segment["frames"][-1])]["file_path"]
    match = TIMESTAMP.search(path)
    if match is None:
        raise RuntimeError(f"cannot recover nuScenes timestamp from {path}")
    return int(match.group(1))


def _cpu_atom(atom: PlasticityAtom) -> PlasticityAtom:
    return PlasticityAtom(*(value.detach().cpu() for value in (
        atom.xyz, atom.scale, atom.key, atom.code, atom.confidence,
    )))


def _device_atom(atom: PlasticityAtom, device: torch.device) -> PlasticityAtom:
    return PlasticityAtom(*(value.to(device) for value in (
        atom.xyz, atom.scale, atom.key, atom.code, atom.confidence,
    )))


def _float_stats(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach()) for key, value in stats.items()}


def _tensor_stats(stats: dict[str, float], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, device=device) for key, value in stats.items()}


def _memory_state(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    config: dict,
) -> dict:
    zero = segment.atom(head)
    code, _ = adapt_context(
        head, segment, zero.code,
        step_size=float(config["adaptation"]["ttt_step_size"]),
        steps=int(config["adaptation"]["ttt_steps"]),
    )
    pre, pre_stats = geometry_objective(head, segment, zero.code, return_stats=True)
    post, post_stats = geometry_objective(head, segment, code, return_stats=True)
    atom = replace(zero, code=code.detach())
    return {
        "atom": _cpu_atom(atom),
        "descriptor": atom.key.mean(dim=(1, 2))[0].detach().cpu(),
        "pre": float(pre.detach()), "post": float(post.detach()),
        "pre_stats": _float_stats(pre_stats), "post_stats": _float_stats(post_stats),
    }


def _pair_models(config: dict, dino_pair_cache: dict) -> dict[str, object]:
    pairs = json.loads(Path(config["data"]["key_pair_manifest"]).read_text())
    labels = np.asarray([int(row["label"]) for row in pairs], dtype=np.int64)
    locations = np.asarray([row["location"] for row in pairs])
    matrix = []
    for row in pairs:
        left = dino_pair_cache["rows"][_identifier(row["left"])]["dinov2"].float().numpy()
        right = dino_pair_cache["rows"][_identifier(row["right"])]["dinov2"].float().numpy()
        matrix.append(_token_pair_features(left, right))
    matrix = np.asarray(matrix, dtype=np.float64)
    models = {}
    for held_out in sorted(set(locations.tolist())):
        train = locations != held_out
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=int(config["seed"]),
            ),
        )
        model.fit(matrix[train], labels[train])
        models[held_out] = model
    return models


def _router_models(config: dict) -> tuple[dict[str, object], dict[str, float], list[int]]:
    candidate = json.loads(Path(config["data"]["stage5_candidate_cache"]).read_text())
    stage6 = json.loads(Path(config["data"]["stage6_result"]).read_text())
    if not (
        candidate.get("split") == stage6.get("split") == "train"
        and candidate.get("validation_accessed") is False
        and stage6.get("validation_accessed") is False
        and candidate.get("test_accessed") is False
        and stage6.get("test_accessed") is False
        and candidate.get("query_or_future_router_input") is False
        and stage6.get("query_or_future_router_input") is False
    ):
        raise RuntimeError("Stage 7 router requires locked train-only Stage-5/6 artifacts")
    rows = candidate["rows"]
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    utility = np.asarray([row["future_utility"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["component"] for row in rows])
    columns = primary_feature_columns(
        int(config["router"]["descriptor_dimensions"]),
        tuple(config["router"]["primary_scalar_indices"]),
    )
    if tuple(config["router"]["primary_scalar_indices"]) != PRIMARY_SCALAR_INDICES:
        raise RuntimeError("Stage 7 router feature contract changed")
    models, thresholds = {}, {}
    for component in sorted(set(groups.tolist())):
        train = groups != component
        model = _model(config)
        model.fit(matrix[train][:, columns], utility[train])
        models[component] = model
        thresholds[component] = float(
            stage6["threshold_by_component"][component]["threshold"]
        )
    return models, thresholds, columns


def _paired_component_bootstrap(
    left: dict[str, float],
    right: dict[str, float],
    group_by_episode: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict:
    groups = sorted(set(group_by_episode.values()))
    by_group = {
        group: [episode for episode in left if group_by_episode[episode] == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        episodes = [episode for group in chosen for episode in by_group[str(group)]]
        values.append(float(np.mean([left[episode] - right[episode] for episode in episodes])))
    array = np.asarray(values)
    low, high = np.percentile(array, [2.5, 97.5])
    return {
        "unit": "physical_overlap_component", "components": len(groups),
        "samples": samples, "seed": seed,
        "mean_difference": float(array.mean()), "ci95": [float(low), float(high)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_causal_dino_retrieval_v17.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 causal retrieval requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    candidate_path = Path(config["output"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 Stage-7 output already exists")
    if config["data"]["split"] != "train":
        raise RuntimeError("EXP-009 Stage 7 is train-only")

    manifest = json.loads(Path(config["data"]["geometry_manifest"]).read_text())
    geometry = torch.load(
        config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True,
    )
    dino = torch.load(
        config["output"]["dinov2_context_cache"], map_location="cpu", weights_only=False,
    )
    dino_pair = torch.load(
        config["data"]["dinov2_pair_cache"], map_location="cpu", weights_only=False,
    )
    stage4 = json.loads(Path(config["data"]["stage4_result"]).read_text())
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 225
        and geometry.get("split") == dino.get("split") == dino_pair.get("split") == "train"
        and dino.get("contexts") == 557
        and dino.get("validation_accessed") is False
        and dino.get("test_accessed") is False
        and stage4.get("selected_consolidation_representation") == "dinov2"
    ):
        raise RuntimeError("Stage-7 train/cache/selection contract failed")

    context_info: dict[str, dict] = {}
    targets: dict[str, dict] = {}
    for index, row in enumerate(manifest):
        for tag, cache_tag in (
            ("a", "a_context"), ("b", "b_context"), ("a_prime", "a_prime_context"),
        ):
            segment = row[tag]
            key = _identifier(segment)
            info = {
                "id": key, "segment": segment, "cache_index": index, "cache_tag": cache_tag,
                "scene": segment["scene"], "location": row["location"],
            }
            if key in context_info:
                previous = context_info[key]
                if (
                    previous["segment"] != segment
                    or previous["scene"] != info["scene"]
                    or previous["location"] != info["location"]
                ):
                    raise RuntimeError("duplicate context has inconsistent metadata")
            else:
                context_info[key] = info
        target_key = _identifier(row["a_prime"])
        target = {
            "id": target_key, "cache_index": index,
            "episode": f"target-{target_key}",
            "component": f"component-{int(row['component_id'])}",
            "location": row["location"], "matched_context": _identifier(row["a"]),
            "query_frames": tuple(int(value) for value in row["a_prime"]["query_frames"]),
        }
        if target_key in targets:
            previous = targets[target_key]
            if not (
                previous["component"] == target["component"]
                and previous["location"] == target["location"]
                and previous["query_frames"] == target["query_frames"]
            ):
                raise RuntimeError("duplicate target has inconsistent query/group metadata")
        else:
            targets[target_key] = target
    if len(context_info) != 557 or len(targets) != 218:
        raise RuntimeError("unexpected unique context/target count")
    if set(context_info) != set(dino["rows"]):
        raise RuntimeError("DINO context cache does not exactly cover the causal stream")

    scene_root = Path(config["data"]["scene_root"])
    metadata_cache: dict[str, dict] = {}
    for info in context_info.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)
    events = sorted(context_info.values(), key=lambda row: (row["timestamp"], row["id"]))

    atom_checkpoint = torch.load(
        config["models"]["plasticity_atom_checkpoint"], map_location="cpu", weights_only=False,
    )
    if not (
        atom_checkpoint.get("protocol_revision") == "v2.7"
        and atom_checkpoint.get("split") == "train"
        and atom_checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("Stage 7 requires the locked EXP-006 atom")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["models"]["feature_dim"])).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    pair_models = _pair_models(config, dino_pair)
    router_models, router_thresholds, router_columns = _router_models(config)

    policies = tuple(config["retrieval"]["policies"])
    candidate_count = int(config["retrieval"]["candidate_count"])
    strength = float(config["adaptation"]["reuse_strength"])
    epsilon = float(config["adaptation"]["utility_deadband"])
    memory: dict[str, dict] = {}
    bank: list[str] = []
    candidate_rows = []
    selection_rows = []
    oracle_values = {policy: {} for policy in policies}
    oracle_accept = {policy: {} for policy in policies}
    router_values = {policy: {} for policy in policies}
    router_accept = {policy: {} for policy in policies}
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
                step_size=float(config["adaptation"]["ttt_step_size"]),
                steps=int(config["adaptation"]["ttt_steps"]),
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
                current_descriptor = zero.key.mean(dim=(1, 2))[0]
                query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                query = CachedAtomSegment.from_cache(query_payload, "query", device)
                current_atom = replace(zero, code=code)
                current_query = query_readout_loss(head, current_atom, query)

                policy_candidates: dict[str, list[str]] = {policy: [] for policy in policies}
                policy_scores: dict[str, dict[str, float]] = {policy: {} for policy in policies}
                if bank:
                    current_dino = dino["rows"][key]["dinov2"].float().numpy()
                    dino_features = np.asarray([
                        _token_pair_features(
                            current_dino, dino["rows"][candidate]["dinov2"].float().numpy(),
                        ) for candidate in bank
                    ], dtype=np.float64)
                    dino_score = pair_models[target["location"]].predict_proba(dino_features)[:, 1]
                    vggt_score = np.asarray([
                        float(torch.nn.functional.cosine_similarity(
                            state["descriptor"], memory[candidate]["descriptor"], dim=0,
                        )) for candidate in bank
                    ])
                    ranked_dino = sorted(
                        zip(bank, dino_score.tolist()), key=lambda row: (-row[1], row[0]),
                    )[:candidate_count]
                    ranked_vggt = sorted(
                        zip(bank, vggt_score.tolist()), key=lambda row: (-row[1], row[0]),
                    )[:candidate_count]
                    policy_candidates["dinov2"] = [row[0] for row in ranked_dino]
                    policy_candidates["vggt_transport_key"] = [row[0] for row in ranked_vggt]
                    policy_scores["dinov2"] = dict(ranked_dino)
                    policy_scores["vggt_transport_key"] = dict(ranked_vggt)
                    policy_candidates["fifo"] = bank[-candidate_count:][::-1]
                    policy_scores["fifo"] = {
                        candidate: float(rank) for rank, candidate in enumerate(
                            policy_candidates["fifo"][::-1], start=1,
                        )
                    }
                    stable_seed = int(hashlib.sha1(episode.encode()).hexdigest()[:8], 16)
                    generator = random.Random(int(config["seed"]) + stable_seed)
                    sample = generator.sample(bank, min(candidate_count, len(bank)))
                    policy_candidates["deterministic_random"] = sample
                    policy_scores["deterministic_random"] = {
                        candidate: float(-rank) for rank, candidate in enumerate(sample)
                    }

                union = sorted({candidate for values in policy_candidates.values() for candidate in values})
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
                        current_descriptor=current_descriptor,
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
                        "target_context": key, "candidate_context": candidate,
                        "candidate_scene": context_info[candidate]["scene"],
                        "future_utility": utility, "predicted_utility": prediction,
                        "features": [float(value) for value in features.detach().cpu()],
                    })

                selected = {"episode": episode, "component": target["component"],
                            "target_context": key, "bank_size": len(bank), "policies": {}}
                for policy in policies:
                    candidates = policy_candidates[policy]
                    if candidates:
                        oracle_choice = max(candidates, key=lambda value: evaluated[value]["utility"])
                        oracle_utility = max(0.0, evaluated[oracle_choice]["utility"])
                        router_choice = max(candidates, key=lambda value: evaluated[value]["prediction"])
                        router_score = evaluated[router_choice]["prediction"]
                        take = router_score > router_thresholds[target["component"]]
                        router_utility = evaluated[router_choice]["utility"] if take else 0.0
                    else:
                        oracle_choice = router_choice = None
                        oracle_utility = router_utility = router_score = 0.0
                        take = False
                    oracle_values[policy][episode] = float(oracle_utility)
                    oracle_accept[policy][episode] = oracle_utility > 0.0
                    router_values[policy][episode] = float(router_utility)
                    router_accept[policy][episode] = bool(take)
                    selected["policies"][policy] = {
                        "candidates": candidates,
                        "retrieval_scores": policy_scores[policy],
                        "oracle_candidate": oracle_choice, "oracle_utility": float(oracle_utility),
                        "router_candidate": router_choice, "router_score": float(router_score),
                        "router_accepted": bool(take), "router_utility": float(router_utility),
                        "matched_in_topk": target["matched_context"] in candidates,
                    }
                selection_rows.append(selected)

            memory[key] = state
            bank.append(key)
            if (event_index + 1) % 25 == 0 or event_index + 1 == len(events):
                print(json.dumps({
                    "events": event_index + 1, "total": len(events),
                    "targets_evaluated": len(selection_rows), "bank": len(bank),
                }), flush=True)

    expected_episodes = set(group_by_episode)
    if any(set(values) != expected_episodes for values in oracle_values.values()):
        raise RuntimeError("Stage-7 oracle policy did not cover every unique target")
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
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    bootstrap = {
        "dinov2_minus_vggt_oracle_topk": _paired_component_bootstrap(
            oracle_values["dinov2"], oracle_values["vggt_transport_key"], group_by_episode,
            samples=samples, seed=seed,
        ),
        "dinov2_minus_vggt_router": _paired_component_bootstrap(
            router_values["dinov2"], router_values["vggt_transport_key"], group_by_episode,
            samples=samples, seed=seed + 1,
        ),
    }
    checks = {
        "dinov2_oracle_topk_beats_vggt":
            metrics["dinov2"]["oracle_topk"]["mean_selected_utility"]
            > metrics["vggt_transport_key"]["oracle_topk"]["mean_selected_utility"],
        "dinov2_router_beats_vggt":
            metrics["dinov2"]["router"]["mean_selected_utility"]
            > metrics["vggt_transport_key"]["router"]["mean_selected_utility"],
        "dinov2_router_harm_not_above_vggt":
            metrics["dinov2"]["router"]["harmful_rate"]
            <= metrics["vggt_transport_key"]["router"]["harmful_rate"],
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "stage": "stage7_causal_dino_candidates",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "rows": candidate_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage7_causal_dino_retrieval",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "contexts": len(events), "unique_targets": len(targets),
        "components": len(set(group_by_episode.values())),
        "timestamp_order": True, "unique_context_writes": True,
        "candidate_count": candidate_count, "candidate_evaluations": len(candidate_rows),
        "metrics": metrics, "bootstrap": bootstrap, "selection_rows": selection_rows,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "metrics": metrics,
        "bootstrap": bootstrap, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
