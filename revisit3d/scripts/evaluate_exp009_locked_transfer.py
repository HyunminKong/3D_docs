#!/usr/bin/env python3
"""Evaluate the fully locked EXP-006 local-reuse path on unseen EXP-009 train scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    PRIMARY_SCALAR_INDICES,
    adapt_context,
    geometry_objective,
    observable_router_features,
    primary_feature_columns,
    query_readout_loss,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.evaluate_exp006_validation import _bootstrap, _summarize
from revisit3d.scripts.train_exp006_atom import _segments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_locked_transfer_v15.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 locked transfer requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    result_path = Path(config["stage2"]["result"])
    candidate_path = Path(config["stage2"]["candidate_cache"])
    if result_path.exists() or candidate_path.exists():
        raise RuntimeError("EXP-009 locked-transfer output already exists")

    cache_path = Path(config["stage1"]["cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not (
        cache.get("protocol_revision") == config["protocol_revision"]
        and cache.get("split") == "train"
        and len(cache.get("rows", [])) == 225
    ):
        raise RuntimeError("EXP-009 transfer cache violates the train-only v1.5 contract")
    atom_path = Path(config["stage1"]["source_checkpoint"])
    atom_checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    if not (
        atom_checkpoint.get("protocol_revision") == "v2.7"
        and atom_checkpoint.get("split") == "train"
        and atom_checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("source atom is not the locked EXP-006 train model")
    router_path = Path(config["stage2"]["locked_router_model"])
    router_result = json.loads(Path(config["stage2"]["locked_router_result"]).read_text())
    router_payload = joblib.load(router_path)
    expected_columns = primary_feature_columns(
        int(config["stage2"]["descriptor_dimensions"]),
        tuple(config["stage2"]["primary_scalar_indices"]),
    )
    if not (
        router_result.get("validation_accessed") is False
        and router_result.get("test_accessed") is False
        and router_result.get("query_or_future_router_input") is False
        and router_result.get("model_sha256") == _sha256(router_path)
        and router_payload.get("protocol_revision") == "v2.8"
        and router_payload.get("feature_columns") == expected_columns
        and tuple(config["stage2"]["primary_scalar_indices"]) == PRIMARY_SCALAR_INDICES
    ):
        raise RuntimeError("locked router artifact or feature contract changed")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    records = dataset.records
    if len(records) != 225:
        raise RuntimeError("locked transfer requires the frozen 225-episode pilot")
    group_by_episode = {
        row["episode_id"]: f"component-{int(row['component_id'])}" for row in records
    }
    if len(set(group_by_episode.values())) < int(config["success"]["minimum_physical_components"]):
        raise RuntimeError("geometry pilot has too few physical components")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    strength = float(stage1["reuse_strength"])
    epsilon = float(stage1["utility_deadband_minimum"])
    candidate_rows = []
    episode_aux = {}

    with torch.enable_grad():
        for index, record in enumerate(records):
            current, query, sources = _segments(cache, records, index, config, device)
            current_zero = current.atom(head)
            base_query = query_readout_loss(head, current_zero, query)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            current_atom = replace(current_zero, code=current_code)
            current_query = query_readout_loss(head, current_atom, query)
            current_pre, current_pre_stats = geometry_objective(
                head, current, current_zero.code, return_stats=True,
            )
            current_post, current_post_stats = geometry_objective(
                head, current, current_code, return_stats=True,
            )
            current_descriptor = current_zero.key.mean(dim=(1, 2))[0]
            episode = record["episode_id"]
            visual_codes = []
            for label, source in sources:
                source_zero = source.atom(head)
                source_code, _ = adapt_context(
                    head, source, source_zero.code,
                    step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
                )
                source_pre, source_pre_stats = geometry_objective(
                    head, source, source_zero.code, return_stats=True,
                )
                source_post, source_post_stats = geometry_objective(
                    head, source, source_code, return_stats=True,
                )
                source_atom = replace(source_zero, code=source_code.detach())
                alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
                visual = visual_transport(source_atom, current_zero)
                visual_codes.append(visual.code)
                candidate_code = (current_code + strength * visual.code).clamp(-1, 1)
                candidate_objective = geometry_objective(head, current, candidate_code)
                candidate_query = query_readout_loss(
                    head, replace(current_zero, code=candidate_code), query,
                )
                features = observable_router_features(
                    current_descriptor=current_descriptor,
                    source_descriptor=source_atom.key.mean(dim=(1, 2))[0],
                    current_code=current_code, transported_code=visual.code,
                    visual_result=visual, alignment=alignment,
                    current_pre_objective=current_pre, current_post_objective=current_post,
                    candidate_objective=candidate_objective,
                    source_pre_objective=source_pre, source_post_objective=source_post,
                    current_pre_stats=current_pre_stats, current_post_stats=current_post_stats,
                    source_pre_stats=source_pre_stats, source_post_stats=source_post_stats,
                )
                candidate_rows.append({
                    "episode": episode, "component": group_by_episode[episode],
                    "candidate": label, "features": [float(value) for value in features.detach().cpu()],
                    "future_utility": float(normalized_future_utility(current_query, candidate_query).detach()),
                    "current_objective_improvement": float(features[259].detach()),
                    "appearance_similarity": float(features[267].detach()),
                })
            pooled = torch.stack(visual_codes).mean(0)
            pooled_query = query_readout_loss(
                head, replace(current_zero, code=(current_code + strength * pooled).clamp(-1, 1)), query,
            )
            episode_aux[episode] = {
                "base_query": float(base_query.detach()),
                "current_query": float(current_query.detach()),
                "current_to_base": float((current_query / base_query.detach().abs().clamp_min(1e-6)).detach()),
                "visual_mean_utility": float(normalized_future_utility(current_query, pooled_query).detach()),
            }
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                print(json.dumps({"evaluated": index + 1, "total": len(records)}), flush=True)

    matrix = np.asarray([row["features"] for row in candidate_rows], dtype=np.float64)
    prediction = router_payload["model"].predict(matrix[:, expected_columns])
    for row, value in zip(candidate_rows, prediction):
        row["predicted_utility"] = float(value)

    methods = {name: {} for name in (
        "router", "visual_mean", "current_objective", "appearance_similarity",
        "matched_identity", "random_expectation", "oracle",
    )}
    accepted = {name: {} for name in methods}
    selection_rows = []
    threshold = float(router_payload["utility_threshold"])
    for episode in sorted(episode_aux):
        subset = [row for row in candidate_rows if row["episode"] == episode]
        utility = np.asarray([row["future_utility"] for row in subset])

        def choose(name: str, scores: np.ndarray, gated: bool) -> int:
            choice = int(scores.argmax())
            take = bool(scores[choice] > threshold) if gated else True
            methods[name][episode] = float(utility[choice]) if take else 0.0
            accepted[name][episode] = take
            return choice

        router_choice = choose("router", np.asarray([row["predicted_utility"] for row in subset]), True)
        choose("current_objective", np.asarray([row["current_objective_improvement"] for row in subset]), True)
        choose("appearance_similarity", np.asarray([row["appearance_similarity"] for row in subset]), False)
        matched = next(index for index, row in enumerate(subset) if row["candidate"] == "matched_a")
        methods["matched_identity"][episode] = float(utility[matched])
        accepted["matched_identity"][episode] = True
        methods["random_expectation"][episode] = float(utility.mean())
        accepted["random_expectation"][episode] = True
        methods["visual_mean"][episode] = episode_aux[episode]["visual_mean_utility"]
        accepted["visual_mean"][episode] = True
        methods["oracle"][episode] = max(0.0, float(utility.max()))
        accepted["oracle"][episode] = methods["oracle"][episode] > 0
        selection_rows.append({
            "episode": episode, "component": group_by_episode[episode],
            "router_candidate": subset[router_choice]["candidate"] if accepted["router"][episode] else None,
            **{f"{name}_utility": values[episode] for name, values in methods.items()},
        })

    metrics = {
        name: _summarize(values, accepted[name], group_by_episode, epsilon)
        for name, values in methods.items()
    }
    bootstrap = _bootstrap(methods, group_by_episode, int(config["statistics"]["bootstrap_samples"]),
                           int(config["statistics"]["bootstrap_seed"]))
    mean_current_to_base = float(np.mean([row["current_to_base"] for row in episode_aux.values()]))
    checks = {
        "current_objective_healthy": mean_current_to_base
        <= float(config["success"]["maximum_mean_current_to_base_ratio"]),
        "visual_utility_positive": metrics["visual_mean"]["mean_selected_utility"]
        > float(config["success"]["minimum_visual_mean_utility"]),
        "visual_harm_bounded": metrics["visual_mean"]["harmful_rate"]
        < float(config["success"]["maximum_visual_deadband_harm"]),
        "router_beats_visual": metrics["router"]["mean_selected_utility"]
        > metrics["visual_mean"]["mean_selected_utility"],
        "router_harm_not_above_visual": metrics["router"]["harmful_rate"]
        <= metrics["visual_mean"]["harmful_rate"],
        "component_health": len(set(group_by_episode.values()))
        >= int(config["success"]["minimum_physical_components"]),
    }
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({
        "experiment": "EXP-009", "protocol_revision": config["protocol_revision"],
        "split": "train", "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "rows": candidate_rows,
    }, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage5_locked_local_reuse_transfer",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False,
        "source_atom": str(atom_path), "source_router": str(router_path),
        "episodes": len(records), "components": len(set(group_by_episode.values())),
        "candidates": len(candidate_rows), "mean_current_to_base_ratio": mean_current_to_base,
        "metrics": metrics, "bootstrap": bootstrap,
        "selection_rows": selection_rows,
        "candidate_cache": str(candidate_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "mean_current_to_base": mean_current_to_base,
        "metrics": metrics, "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
