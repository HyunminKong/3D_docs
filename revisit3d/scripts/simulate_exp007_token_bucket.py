#!/usr/bin/env python3
"""Token-set OOF bucket prediction and causal consolidation for EXP-007."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from revisit3d.experiments import grouped_folds, require_exp006_split
from revisit3d.models import SpatialPlasticityHead
from revisit3d.scripts.simulate_exp007_causal_bank import _evaluate, _orders, _sha256
from revisit3d.scripts.simulate_exp007_utility_consolidation import (
    _score,
    _summary,
    _update_history,
)


def _base_policy(policy: str) -> str:
    return {
        "token_bucket_predicted_history": "predicted_history",
        "token_bucket_delayed_topk_utility": "delayed_topk_utility",
    }[policy]


def _token_pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    similarity = left @ right.T
    left_max = similarity.max(axis=1)
    right_max = similarity.max(axis=0)
    right_for_left = similarity.argmax(axis=1)
    left_for_right = similarity.argmax(axis=0)
    left_index = np.arange(len(left))
    mutual = left_for_right[right_for_left] == left_index
    mutual_score = similarity[left_index[mutual], right_for_left[mutual]]
    pooled_left = left.mean(axis=0)
    pooled_right = right.mean(axis=0)
    pooled_cosine = float(
        pooled_left @ pooled_right
        / max(np.linalg.norm(pooled_left) * np.linalg.norm(pooled_right), 1e-12)
    )
    return np.asarray([
        pooled_cosine,
        float(left_max.mean()),
        float(right_max.mean()),
        float(np.quantile(np.concatenate((left_max, right_max)), 0.10)),
        float(np.quantile(np.concatenate((left_max, right_max)), 0.90)),
        float(0.5 * ((left_max >= 0.60).mean() + (right_max >= 0.60).mean())),
        float(mutual.mean()),
        float(mutual_score.mean()) if mutual_score.size else 0.0,
    ])


def _crossfit_pca(
    cache: dict, held_out: list[int], *, components: int, sample_size: int,
    iterations: int, seed: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a fold-local projection without reading held-out episode features."""
    held_out_set = set(held_out)
    context_tags = ("a_context", "b_context", "a_prime_context")
    matrix = torch.cat([
        row["segments"][tag]["features"].flatten(0, 2)
        for index, row in enumerate(cache["rows"])
        if index not in held_out_set
        for tag in context_tags
    ], dim=0)
    generator = torch.Generator().manual_seed(seed)
    selection = torch.randperm(matrix.shape[0], generator=generator)[:min(sample_size, matrix.shape[0])]
    sample = matrix[selection].to(device=device, dtype=torch.float32)
    sample = F.layer_norm(sample, (sample.shape[-1],))
    mean = sample.mean(dim=0)
    torch.manual_seed(seed)
    _, _, vectors = torch.pca_lowrank(
        sample - mean, q=components, center=False, niter=iterations,
    )
    return mean, vectors.transpose(0, 1).contiguous()


def _write_token(
    bank: list[dict], context: dict, policy: str, capacity: int, serial: int,
    probability: dict[tuple[str, str], float], threshold: float, config: dict, counts: dict,
) -> None:
    counts["writes"] += 1
    merge_index = None
    if bank:
        scores = [
            1.0 if entry["context_id"] == context["context_id"] else probability[
                tuple(sorted((entry["context_id"], context["context_id"])))
            ]
            for entry in bank
        ]
        best = int(np.argmax(scores))
        if scores[best] >= threshold:
            merge_index = best
    if merge_index is not None:
        entry = bank[merge_index]
        if context["scene"] in entry["diagnostic_scenes"]:
            counts["true_bucket_merges"] += 1
        else:
            counts["false_bucket_merges"] += 1
        entry.update({
            "context_id": context["context_id"],
            "descriptor": context["descriptor"],
            "serial": serial,
            "frequency": entry["frequency"] + 1,
            "diagnostic_scenes": sorted(set(entry["diagnostic_scenes"] + [context["scene"]])),
        })
        counts["merges"] += 1
        return
    entry = {
        "context_id": context["context_id"],
        "scene": context["scene"],  # diagnostic only
        "diagnostic_scenes": [context["scene"]],
        "descriptor": context["descriptor"],
        "serial": serial,
        "frequency": 1,
        "pred_sum": 0.0,
        "pred_count": 0,
        "utility_sum": 0.0,
        "utility_count": 0,
    }
    candidates = bank + [entry]
    if len(candidates) > capacity:
        base = _base_policy(policy)
        remove = min(
            range(len(candidates)),
            key=lambda index: (_score(candidates[index], base, config), candidates[index]["serial"]),
        )
        if remove == len(bank):
            counts["rejected_writes"] += 1
        else:
            counts["evictions"] += 1
        candidates.pop(remove)
    bank[:] = candidates


def _bootstrap(
    left: dict[str, float], right: dict[str, float], groups: list[str], samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        draws.append(float(np.mean([left[str(group)] - right[str(group)] for group in sampled])))
    array = np.asarray(draws)
    low, high = np.percentile(array, [2.5, 97.5])
    return {"bootstrap_mean": float(array.mean()), "ci95": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_token_bucket_v18.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("token-bucket evaluation requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    require_exp006_split(config["data"]["split"])
    output = Path(config["bucket_model"]["result"])
    if output.exists():
        raise RuntimeError(f"token-bucket result already exists: {output}")
    table_path = Path(config["source"]["oof_utility_table"])
    table = json.loads(table_path.read_text())
    cache_path = Path(config["source"]["cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    capacity_path = Path(config["source"]["oof_capacity_result"])
    capacity_result = json.loads(capacity_path.read_text())
    oracle_scene_path = Path(config["source"]["oracle_scene_result"])
    oracle_scene = json.loads(oracle_scene_path.read_text())
    if not (
        cache.get("split") == table.get("split") == "train"
        and table.get("validation_accessed") is False
        and capacity_result.get("validation_accessed") is False
        and oracle_scene.get("validation_accessed") is False
        and table.get("cross_fold_bank_mixing") is False
    ):
        raise RuntimeError("token-bucket evaluation requires train-only fold-local sources")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    records = [row for row in manifest if row["split"] == "train"]
    _, group_list = grouped_folds(records, folds=5, seed=600)
    group_by_episode = {
        record["episode_id"]: group_list[index] for index, record in enumerate(records)
    }

    device = torch.device("cuda")
    pair_rows = []
    token_by_fold = {}
    key_source = config["bucket_model"].get("key_source", "learned_atom_key")
    for stream in table["streams"]:
        fold = int(stream["fold"])
        if key_source == "learned_atom_key":
            checkpoint = torch.load(stream["atom_checkpoint"], map_location="cpu", weights_only=False)
            head = SpatialPlasticityHead(feature_dim=2048).to(device)
            head.load_state_dict(checkpoint["head"])
            head.eval().requires_grad_(False)
        elif key_source == "frozen_foundation_train_pca":
            head = None
            pca_mean = cache["pca_mean"].to(device=device, dtype=torch.float32)
            pca_components = cache["pca_components"].to(device=device, dtype=torch.float32)
        elif key_source == "frozen_foundation_crossfit_pca":
            head = None
            pca_mean, pca_components = _crossfit_pca(
                cache,
                [int(index) for index in stream["held_out"]],
                components=int(config["bucket_model"]["pca_components"]),
                sample_size=int(config["bucket_model"]["pca_sample_size"]),
                iterations=int(config["bucket_model"]["pca_iterations"]),
                seed=int(config["seed"]) + fold,
                device=device,
            )
        else:
            raise RuntimeError(f"unsupported bucket key source {key_source!r}")
        token_by_context = {}
        with torch.no_grad():
            for context in stream["contexts"]:
                occurrence = context["observations"][0]
                payload = cache["rows"][occurrence["episode_index"]]["segments"][occurrence["tag"]]
                features = payload["features"].to(device=device, dtype=torch.float32)
                if head is not None:
                    key = head.appearance_key(features).mean(dim=1)[0]
                else:
                    normalized = F.layer_norm(features, (features.shape[-1],))
                    key = ((normalized - pca_mean) @ pca_components.transpose(0, 1)).mean(dim=1)[0]
                key = F.normalize(key, dim=-1)
                token_by_context[context["context_id"]] = key.cpu().numpy()
        token_by_fold[fold] = token_by_context
        for left, right in itertools.combinations(stream["contexts"], 2):
            pair_rows.append({
                "fold": fold,
                "left": left["context_id"],
                "right": right["context_id"],
                "same_scene": left["scene"] == right["scene"],
                "features": [float(value) for value in _token_pair_features(
                    token_by_context[left["context_id"]], token_by_context[right["context_id"]],
                )],
            })
        if head is not None:
            del head
        torch.cuda.empty_cache()

    pair_matrix = np.asarray([row["features"] for row in pair_rows], dtype=np.float64)
    pair_label = np.asarray([row["same_scene"] for row in pair_rows], dtype=bool)
    pair_fold = np.asarray([row["fold"] for row in pair_rows])
    pair_probability = np.empty(len(pair_rows))
    fold_metrics = []
    for held_out in sorted(set(pair_fold.tolist())):
        train, test = pair_fold != held_out, pair_fold == held_out
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(config["bucket_model"]["regularization_c"]),
                class_weight=config["bucket_model"]["class_weight"],
                max_iter=2000,
                random_state=int(config["seed"]),
            ),
        )
        model.fit(pair_matrix[train], pair_label[train])
        pair_probability[test] = model.predict_proba(pair_matrix[test])[:, 1]
        prediction = pair_probability[test] >= float(config["bucket_model"]["probability_threshold"])
        label = pair_label[test]
        fold_metrics.append({
            "held_out_fold": held_out,
            "train_pairs": int(train.sum()),
            "test_pairs": int(test.sum()),
            "test_positive_pairs": int(label.sum()),
            "roc_auc": float(roc_auc_score(label, pair_probability[test])),
            "balanced_accuracy": float(0.5 * (
                prediction[label].mean() + (~prediction[~label]).mean()
            )),
            "precision": float(label[prediction].mean()) if prediction.any() else 1.0,
            "same_scene_recall": float(prediction[label].mean()),
            "different_scene_rejection": float((~prediction[~label]).mean()),
        })
    probability_by_fold = {fold: {} for fold in sorted(set(pair_fold.tolist()))}
    for row, probability in zip(pair_rows, pair_probability):
        probability_by_fold[row["fold"]][tuple(sorted((row["left"], row["right"])))] = float(probability)

    reference_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "unbounded_all_write" and row["capacity"] is None
    ]
    oracle_reference = {
        (row["fold"], row["order"], row["episode"]): row["causal_unbounded_oracle"]
        for row in reference_rows
    }
    capacity = int(config["bank"]["capacity"])
    top_k = int(config["bank"]["top_k"])
    threshold = float(config["bucket_model"]["probability_threshold"])
    variants = []
    all_rows = []
    for policy in config["bank"]["policies"]:
        rows = []
        counts = {
            "writes": 0, "merges": 0, "evictions": 0, "rejected_writes": 0,
            "true_bucket_merges": 0, "false_bucket_merges": 0,
        }
        for stream in table["streams"]:
            fold = int(stream["fold"])
            contexts = {row["context_id"]: row for row in stream["contexts"]}
            lookup = {(row["episode"], row["context_id"]): row for row in stream["pairs"]}
            orders = _orders(config["bank"]["stream_orders"], stream["events"])
            for order_index, (order_name, sequence) in enumerate(orders.items()):
                bank = []
                serial = int(config["seed"]) + 100000 * fold + 1000 * order_index
                for position, event_index in enumerate(sequence):
                    event = stream["events"][event_index]
                    for context_id in event["pre_query_writes"]:
                        serial += 1
                        _write_token(
                            bank, contexts[context_id], policy, capacity, serial,
                            probability_by_fold[fold], threshold, config["bank"], counts,
                        )
                    evaluation = _evaluate(bank, event["episode"], lookup, top_k, 0.0)
                    evaluation["causal_unbounded_oracle"] = oracle_reference[
                        (fold, order_name, event["episode"])
                    ]
                    rows.append({
                        "fold": fold, "policy": policy, "capacity": capacity,
                        "order": order_name, "stream_position": position,
                        "episode": event["episode"], **evaluation,
                    })
                    _update_history(
                        bank, event["episode"], lookup, _base_policy(policy), top_k,
                        evaluation["selected_context"],
                    )
                    for context_id in event["post_query_writes"]:
                        serial += 1
                        _write_token(
                            bank, contexts[context_id], policy, capacity, serial,
                            probability_by_fold[fold], threshold, config["bank"], counts,
                        )
        summary = _summary(rows, float(table["utility_deadband"]), group_by_episode)
        variants.append({"policy": policy, "capacity": capacity, "summary": summary, "counts": counts})
        all_rows.extend(rows)
        print(json.dumps({"policy": policy, "summary": summary, "counts": counts}), flush=True)

    appearance_variant = next(
        row for row in capacity_result["variants"]
        if row["policy"] == "appearance_diversity" and row["capacity"] == capacity
    )
    appearance = appearance_variant["summary"]["metrics"]["router_topk"]
    appearance_rows = [
        row for row in capacity_result["rows"]
        if row["policy"] == "appearance_diversity" and row["capacity"] == capacity
    ]
    oracle_variant = next(
        row for row in oracle_scene["variants"]
        if row["policy"] == "scene_delayed_topk_utility"
    )
    oracle_metric = oracle_variant["summary"]
    groups = sorted(set(group_list))
    appearance_component = {
        group: float(np.mean([
            row["router_topk"] for row in appearance_rows if group_by_episode[row["episode"]] == group
        ])) for group in groups
    }
    for variant in variants:
        method_rows = [row for row in all_rows if row["policy"] == variant["policy"]]
        method_component = {
            group: float(np.mean([
                row["router_topk"] for row in method_rows if group_by_episode[row["episode"]] == group
            ])) for group in groups
        }
        variant["minus_appearance_component_bootstrap"] = _bootstrap(
            method_component, appearance_component, groups,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["bootstrap_seed"]),
        )
        metric = variant["summary"]
        variant["oracle_scene_utility_retention"] = metric["mean_utility"] / oracle_metric["mean_utility"]
        variant["registered_pass"] = (
            metric["mean_utility"] > appearance["mean_utility"]
            and metric["harmful_rate"] <= appearance["harmful_rate"]
            and variant["oracle_scene_utility_retention"]
            >= float(config["success"]["minimum_oracle_scene_utility_retention"])
        )
    result = {
        "experiment": "EXP-007",
        "stage": (
            "stage8_crossfit_frozen_token_bucket_consolidation"
            if key_source == "frozen_foundation_crossfit_pca"
            else "stage7_frozen_token_bucket_consolidation"
            if key_source == "frozen_foundation_train_pca"
            else "stage6_token_set_bucket_consolidation"
        ),
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "same_event_future_in_features": False,
        "ground_truth_scene_runtime_input": False,
        "ground_truth_scene_role": "crossfit_classifier_target_and_diagnostic_only",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_table": str(table_path),
        "bucket_feature_contract": [
            "pooled_cosine", "left_nn_mean", "right_nn_mean", "nn_q10", "nn_q90",
            "similarity_coverage", "mutual_fraction", "mutual_similarity_mean",
        ],
        "bucket_key_source": key_source,
        "bucket_folds": fold_metrics,
        "bucket_overall_roc_auc": float(roc_auc_score(pair_label, pair_probability)),
        "appearance_diversity_capacity8": appearance,
        "oracle_scene_delayed_topk": oracle_metric,
        "variants": variants,
        "registered_gate": {
            "passed": any(row["registered_pass"] for row in variants),
            "passing_policies": [row["policy"] for row in variants if row["registered_pass"]],
        },
        "pair_rows": [{**row, "oof_probability": float(pair_probability[index])}
                      for index, row in enumerate(pair_rows)],
        "rows": all_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "bucket_auc": result["bucket_overall_roc_auc"],
        "bucket_folds": fold_metrics, "appearance": appearance, "oracle_scene": oracle_metric,
        "gate": result["registered_gate"],
        "variants": [{
            "policy": row["policy"], "summary": row["summary"],
            "retention": row["oracle_scene_utility_retention"],
            "difference": row["minus_appearance_component_bootstrap"],
        } for row in variants],
    }), flush=True)


if __name__ == "__main__":
    main()
