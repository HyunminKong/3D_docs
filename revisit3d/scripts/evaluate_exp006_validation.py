#!/usr/bin/env python3
"""One-shot evaluation of the D023-locked EXP-006 model on validation."""

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
    grouped_folds,
    observable_router_features,
    primary_feature_columns,
    query_readout_loss,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(
    values: dict[str, float], accepted: dict[str, bool], group_by_episode: dict[str, str], epsilon: float,
) -> dict:
    ordered = sorted(values)
    array = np.asarray([values[episode] for episode in ordered], dtype=np.float64)
    group_means = {
        group: float(np.mean([values[episode] for episode in ordered if group_by_episode[episode] == group]))
        for group in sorted(set(group_by_episode.values()))
    }
    group_array = np.asarray(list(group_means.values()), dtype=np.float64)
    return {
        "episodes": len(ordered),
        "mean_selected_utility": float(array.mean()),
        "median_selected_utility": float(np.median(array)),
        "beneficial_rate": float(np.mean(array > epsilon)),
        "harmful_rate": float(np.mean(array < -epsilon)),
        "raw_sign_harm_rate": float(np.mean(array < 0.0)),
        "accept_rate": float(np.mean([accepted[episode] for episode in ordered])),
        "component_mean_utility": group_means,
        "component_harm_rate": float(np.mean(group_array < -epsilon)),
        "harmful_components": [group for group, value in group_means.items() if value < -epsilon],
    }


def _bootstrap(
    methods: dict[str, dict[str, float]], group_by_episode: dict[str, str], samples: int, seed: int,
) -> dict:
    groups = sorted(set(group_by_episode.values()))
    episodes_by_group = {
        group: [episode for episode in group_by_episode if group_by_episode[episode] == group]
        for group in groups
    }
    rng = np.random.default_rng(seed)
    draws = {method: [] for method in methods}
    comparisons = {
        f"router_minus_{baseline}": []
        for baseline in methods if baseline not in ("router", "oracle")
    }
    for _ in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_episodes = [
            episode for group in sampled_groups for episode in episodes_by_group[str(group)]
        ]
        statistic = {
            method: float(np.mean([values[episode] for episode in sampled_episodes]))
            for method, values in methods.items()
        }
        for method, value in statistic.items():
            draws[method].append(value)
        for name in comparisons:
            baseline = name.removeprefix("router_minus_")
            comparisons[name].append(statistic["router"] - statistic[baseline])

    def interval(values: list[float]) -> dict:
        array = np.asarray(values)
        low, high = np.percentile(array, [2.5, 97.5])
        return {"bootstrap_mean": float(array.mean()), "ci95": [float(low), float(high)]}

    return {
        "unit": "physical_overlap_component",
        "components": len(groups),
        "samples": samples,
        "seed": seed,
        "descriptive_only": len(groups) < 10,
        "methods": {name: interval(values) for name, values in draws.items()},
        "differences": {name: interval(values) for name, values in comparisons.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_validation_v28.yaml")
    parser.add_argument(
        "--confirm-one-shot-validation", action="store_true",
        help="Confirm that the locked validation result is being accessed exactly once.",
    )
    args = parser.parse_args()
    if not args.confirm_one_shot_validation:
        raise SystemExit("refusing validation access without --confirm-one-shot-validation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 validation requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"], allow_validation=True)
    if config.get("protocol_revision") != "v2.8" or config["data"]["split"] != "val":
        raise RuntimeError("one-shot evaluator accepts only the locked v2.8 validation config")
    result_path = Path(config["stage2"]["result"])
    if result_path.exists():
        raise RuntimeError(f"one-shot validation result already exists: {result_path}")

    cache_path = Path(config["stage1"]["cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not (
        cache.get("protocol_revision") == "v2.8"
        and cache.get("split") == "val"
        and cache.get("pca_fit_split") == "train"
        and Path(cache.get("pca_source_cache", "")) == Path(config["stage1"]["pca_source_cache"])
    ):
        raise RuntimeError("validation cache violates the train-PCA/split lock")

    atom_path = Path(config["stage1"]["source_checkpoint"])
    atom_checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    if not (
        atom_checkpoint.get("protocol_revision") == config["source_atom_protocol_revision"]
        and atom_checkpoint.get("split") == "train"
        and atom_checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("atom checkpoint is not the locked train-only v2.7 model")

    router_result_path = Path(config["stage2"]["train_result"])
    router_result = json.loads(router_result_path.read_text())
    router_path = Path(config["stage2"]["output_model"])
    if not (
        router_result.get("validation_accessed") is False
        and router_result.get("query_or_future_router_input") is False
        and router_result.get("model_sha256") == _sha256(router_path)
    ):
        raise RuntimeError("locked router artifact hash/contract mismatch")
    router_payload = joblib.load(router_path)
    expected_columns = primary_feature_columns(
        int(config["stage2"]["descriptor_dimensions"]),
        tuple(config["stage2"]["primary_scalar_indices"]),
    )
    if not (
        router_payload.get("protocol_revision") == "v2.8"
        and router_payload.get("feature_columns") == expected_columns
        and tuple(config["stage2"]["primary_scalar_indices"]) == PRIMARY_SCALAR_INDICES
    ):
        raise RuntimeError("router artifact changed after D023 lock")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="val",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    records = dataset.records
    if len(records) != 14 or len(cache["rows"]) != len(records):
        raise RuntimeError("locked validation requires exactly the unchanged 14 episodes")
    _, group_of = grouped_folds(records, folds=2, seed=int(config["seed"]))
    group_by_episode = {
        record.get("episode_id", cache["rows"][index]["episode_id"]): group_of[index]
        for index, record in enumerate(records)
    }

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    stage1 = config["stage1"]
    strength = float(stage1["reuse_strength"])
    epsilon = float(stage1["utility_deadband_minimum"])
    candidate_rows: list[dict] = []
    episode_aux: dict[str, dict] = {}

    with torch.enable_grad():
        for index in range(len(records)):
            current, query, sources = _segments(cache, records, index, config, device)
            current_zero = current.atom(head)
            base_query = query_readout_loss(head, current_zero, query)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
            )
            current_atom = replace(current_zero, code=current_code)
            current_query = query_readout_loss(head, current_atom, query)
            current_pre_objective, current_pre_stats = geometry_objective(
                head, current, current_zero.code, return_stats=True,
            )
            current_post_objective, current_post_stats = geometry_objective(
                head, current, current_code, return_stats=True,
            )
            current_descriptor = current_zero.key.mean(dim=(1, 2))[0]
            episode = records[index].get("episode_id", cache["rows"][index]["episode_id"])
            visual_codes = []
            for label, source_segment in sources:
                source_zero = source_segment.atom(head)
                source_code, _ = adapt_context(
                    head, source_segment, source_zero.code,
                    step_size=float(stage1["ttt_step_size"]), steps=int(stage1["ttt_steps"]),
                )
                source_pre_objective, source_pre_stats = geometry_objective(
                    head, source_segment, source_zero.code, return_stats=True,
                )
                source_post_objective, source_post_stats = geometry_objective(
                    head, source_segment, source_code, return_stats=True,
                )
                source_atom = replace(source_zero, code=source_code.detach())
                alignment = align_atoms(source_atom.detach(), current_zero.detach())[0]
                visual_result = visual_transport(source_atom, current_zero)
                transported_code = visual_result.code
                visual_codes.append(transported_code)
                candidate_code = (current_code + strength * transported_code).clamp(-1, 1)
                candidate_objective = geometry_objective(head, current, candidate_code)
                candidate_query = query_readout_loss(
                    head, replace(current_zero, code=candidate_code), query,
                )
                utility = normalized_future_utility(current_query, candidate_query)
                source_descriptor = source_atom.key.mean(dim=(1, 2))[0]
                features = observable_router_features(
                    current_descriptor=current_descriptor,
                    source_descriptor=source_descriptor,
                    current_code=current_code,
                    transported_code=transported_code,
                    visual_result=visual_result,
                    alignment=alignment,
                    current_pre_objective=current_pre_objective,
                    current_post_objective=current_post_objective,
                    candidate_objective=candidate_objective,
                    source_pre_objective=source_pre_objective,
                    source_post_objective=source_post_objective,
                    current_pre_stats=current_pre_stats,
                    current_post_stats=current_post_stats,
                    source_pre_stats=source_pre_stats,
                    source_post_stats=source_post_stats,
                )
                candidate_rows.append({
                    "episode": episode,
                    "group": group_of[index],
                    "candidate": label,
                    "features": [float(value) for value in features.detach().cpu()],
                    "future_utility": float(utility.detach()),
                    "current_objective_improvement": float(features[259].detach()),
                    "appearance_similarity": float(features[267].detach()),
                    "alignment_valid": bool(alignment.valid),
                    "alignment_inlier_ratio": float(alignment.inlier_ratio),
                    "alignment_residual": (
                        float(alignment.normalized_median_residual) if alignment.valid else None
                    ),
                })
            pooled_code = torch.stack(visual_codes).mean(dim=0)
            pooled_query = query_readout_loss(
                head,
                replace(current_zero, code=(current_code + strength * pooled_code).clamp(-1, 1)),
                query,
            )
            episode_aux[episode] = {
                "base_query": float(base_query.detach()),
                "current_query": float(current_query.detach()),
                "current_to_base": float(
                    (current_query / base_query.detach().abs().clamp_min(1e-6)).detach()
                ),
                "visual_mean_utility": float(normalized_future_utility(current_query, pooled_query).detach()),
            }
            print(json.dumps({"split": "val", "episode": episode, "evaluated": True}), flush=True)

    matrix = np.asarray([row["features"] for row in candidate_rows], dtype=np.float64)
    predictions = router_payload["model"].predict(matrix[:, expected_columns])
    for row, prediction in zip(candidate_rows, predictions):
        row["predicted_utility"] = float(prediction)

    episodes = sorted(episode_aux)
    methods: dict[str, dict[str, float]] = {name: {} for name in (
        "router", "visual_mean", "current_objective", "appearance_similarity",
        "matched_identity", "random_expectation", "oracle",
    )}
    accepted: dict[str, dict[str, bool]] = {name: {} for name in methods}
    regret: dict[str, list[float]] = {name: [] for name in methods}
    selection_rows = []
    threshold = float(router_payload["utility_threshold"])
    for episode in episodes:
        subset = [row for row in candidate_rows if row["episode"] == episode]
        utility = np.asarray([row["future_utility"] for row in subset])
        oracle = max(0.0, float(utility.max()))

        def choose(name: str, score: np.ndarray, *, gate: bool) -> tuple[int, bool, float]:
            choice = int(score.argmax())
            take = bool(score[choice] > threshold) if gate else True
            value = float(utility[choice]) if take else 0.0
            methods[name][episode], accepted[name][episode] = value, take
            return choice, take, value

        router_choice, router_accept, router_value = choose(
            "router", np.asarray([row["predicted_utility"] for row in subset]), gate=True,
        )
        current_choice, current_accept, current_value = choose(
            "current_objective",
            np.asarray([row["current_objective_improvement"] for row in subset]), gate=True,
        )
        appearance_choice, _, appearance_value = choose(
            "appearance_similarity",
            np.asarray([row["appearance_similarity"] for row in subset]), gate=False,
        )
        matched_choice = next(i for i, row in enumerate(subset) if row["candidate"] == "matched_a")
        methods["matched_identity"][episode] = float(utility[matched_choice])
        accepted["matched_identity"][episode] = True
        methods["random_expectation"][episode] = float(utility.mean())
        accepted["random_expectation"][episode] = True
        methods["visual_mean"][episode] = episode_aux[episode]["visual_mean_utility"]
        accepted["visual_mean"][episode] = True
        methods["oracle"][episode] = oracle
        accepted["oracle"][episode] = oracle > 0
        for name in methods:
            regret[name].append(oracle - methods[name][episode])
        selection_rows.append({
            "episode": episode,
            "group": group_by_episode[episode],
            "router_candidate": subset[router_choice]["candidate"] if router_accept else None,
            "router_accepted": router_accept,
            "router_predicted_utility": subset[router_choice]["predicted_utility"],
            "router_future_utility": router_value,
            "current_objective_candidate": subset[current_choice]["candidate"] if current_accept else None,
            "current_objective_future_utility": current_value,
            "appearance_candidate": subset[appearance_choice]["candidate"],
            "appearance_future_utility": appearance_value,
            "matched_future_utility": float(utility[matched_choice]),
            "visual_mean_utility": episode_aux[episode]["visual_mean_utility"],
            "random_expected_utility": float(utility.mean()),
            "oracle_utility": oracle,
        })

    metrics = {
        name: {
            **_summarize(values, accepted[name], group_by_episode, epsilon),
            "mean_future_utility_regret": float(np.mean(regret[name])),
        }
        for name, values in methods.items()
    }
    baselines = (
        "visual_mean", "current_objective", "appearance_similarity",
        "matched_identity", "random_expectation",
    )
    checks = {
        "mean_utility_above_deadband": metrics["router"]["mean_selected_utility"] > epsilon,
        "beats_every_registered_control": all(
            metrics["router"]["mean_selected_utility"] > metrics[name]["mean_selected_utility"]
            for name in baselines
        ),
        "no_harmful_component": metrics["router"]["component_harm_rate"] == 0.0,
        "directional_harm_not_worse_than_visual_mean": (
            metrics["router"]["harmful_rate"] <= metrics["visual_mean"]["harmful_rate"]
        ),
        "nontrivial_acceptance": metrics["router"]["accept_rate"] >= 0.20,
    }
    bootstrap = _bootstrap(
        methods, group_by_episode,
        samples=int(config["statistics"]["bootstrap_samples"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    result = {
        "experiment": "EXP-006",
        "stage": "stage2_one_shot_validation",
        "split": "val",
        "protocol_revision": "v2.8",
        "validation_accessed": True,
        "one_shot": True,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_geometry_role": "evaluation_target_only",
        "predicted_alignment_in_primary_router": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "atom_checkpoint": str(atom_path),
        "atom_checkpoint_sha256": _sha256(atom_path),
        "router_train_result": str(router_result_path),
        "router_model": str(router_path),
        "router_model_sha256": _sha256(router_path),
        "episodes": len(episodes),
        "components": len(set(group_of)),
        "utility_epsilon": epsilon,
        "mean_current_to_base": float(np.mean([
            episode_aux[episode]["current_to_base"] for episode in episodes
        ])),
        "metrics": metrics,
        "decision_rule": {
            "registered_before_validation": True,
            "checks": checks,
            "strict_claim_supported": all(checks.values()),
            "note": "Two validation components permit descriptive feasibility only.",
        },
        "bootstrap": bootstrap,
        "selection_rows": selection_rows,
        "candidate_rows": candidate_rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "result": str(result_path),
        "mean_current_to_base": result["mean_current_to_base"],
        "metrics": metrics,
        "decision_rule": result["decision_rule"],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
