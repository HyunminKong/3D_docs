#!/usr/bin/env python3
"""Out-of-fold EXP-006 v2.6 transport ablation on train components."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    adapt_context,
    geometry_objective,
    query_readout_loss,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, align_atoms, geometry_transport, visual_transport
from revisit3d.scripts.train_exp006_atom import _segments


CONDITIONS = (
    "global_vector", "untransported_local", "visual", "geometry", "geometry_appearance",
    "visual_mean_pool",
)


def _summarize(rows: list[dict], condition: str, epsilon: float) -> dict:
    subset = [row for row in rows if row["condition"] == condition]
    valid = [row for row in subset if row["valid"]]
    values = np.asarray([row["utility"] for row in valid], dtype=np.float64)
    episode_best = []
    for episode in sorted({row["episode"] for row in subset}):
        utility = [row["utility"] for row in valid if row["episode"] == episode]
        episode_best.append(max([0.0, *utility]))
    return {
        "candidates": len(subset), "valid_rate": len(valid) / max(len(subset), 1),
        "mean_utility": float(values.mean()) if values.size else None,
        "median_utility": float(np.median(values)) if values.size else None,
        "beneficial_rate": float(np.mean(values > epsilon)) if values.size else 0.0,
        "harmful_rate": float(np.mean(values < -epsilon)) if values.size else 0.0,
        "episodes_with_beneficial": sum(value > epsilon for value in episode_best),
        "mean_episode_best_utility": float(np.mean(episode_best)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument(
        "--crossfit", default="revisit3d/results/EXP-006/stage1_crossfit_heads_train_v26.json",
    )
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage1_transport_ablation_crossfit_train_v26.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 cross-fit ablation requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache = torch.load(config["stage1"]["cache"], map_location="cpu", weights_only=False)
    crossfit = json.loads(Path(args.crossfit).read_text())
    if not (
        cache.get("protocol_revision") == crossfit.get("protocol_revision") == config["protocol_revision"]
        and cache.get("split") == crossfit.get("split") == "train"
        and crossfit.get("validation_accessed") is False
    ):
        raise RuntimeError("cache/cross-fit protocol mismatch")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="train",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    records = dataset.records
    device = torch.device("cuda")
    stage1 = config["stage1"]
    epsilon = float(stage1["utility_deadband_minimum"])
    strength = float(stage1["reuse_strength"])
    rows: list[dict] = []
    router_features: list[dict] = []
    current_rows: list[dict] = []
    seen: set[int] = set()

    for fold in crossfit["folds"]:
        checkpoint = torch.load(fold["checkpoint"], map_location="cpu", weights_only=False)
        if not (
            checkpoint.get("protocol_revision") == config["protocol_revision"]
            and checkpoint.get("held_out") == fold["held_out"]
            and checkpoint.get("query_readout") == "visual_only"
        ):
            raise RuntimeError(f"cross-fit checkpoint mismatch in fold {fold['fold']}")
        head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
        head.load_state_dict(checkpoint["head"])
        head.eval().requires_grad_(False)
        with torch.enable_grad():
            for index in fold["held_out"]:
                if index in seen:
                    raise RuntimeError(f"episode {index} appears in multiple held-out folds")
                seen.add(index)
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
                current_rows.append({
                    "fold": fold["fold"], "index": index, "episode": episode,
                    "base_query": float(base_query.detach()),
                    "current_query": float(current_query.detach()),
                    "current_to_base": float(
                        (current_query / base_query.detach().abs().clamp_min(1e-6)).detach()
                    ),
                })
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
                    transported = {
                        "global_vector": source_code.detach().mean(dim=(1, 2), keepdim=True).expand_as(current_code),
                        "untransported_local": source_code.detach(),
                        "visual": visual_result.code,
                        "geometry": geometry_transport(
                            source_atom, current_zero, [alignment], appearance_weight=0.0,
                        ).code,
                        "geometry_appearance": geometry_transport(
                            source_atom, current_zero, [alignment],
                            appearance_weight=float(stage1["appearance_weight"]),
                        ).code,
                    }
                    visual_codes.append(transported["visual"])
                    visual_candidate_code = (
                        current_code + strength * transported["visual"]
                    ).clamp(-1, 1)
                    visual_candidate_objective = geometry_objective(
                        head, current, visual_candidate_code,
                    )
                    visual_query_loss = query_readout_loss(
                        head, replace(current_zero, code=visual_candidate_code), query,
                    )
                    visual_utility = normalized_future_utility(current_query, visual_query_loss)
                    source_descriptor = source_atom.key.mean(dim=(1, 2))[0]
                    descriptor = torch.cat((
                        current_descriptor,
                        source_descriptor,
                        current_descriptor - source_descriptor,
                        current_descriptor * source_descriptor,
                    ))
                    denominator = current_pre_objective.detach().abs().clamp_min(1e-6)
                    scalars = torch.stack((
                        current_pre_objective / denominator,
                        current_post_objective / denominator,
                        visual_candidate_objective / denominator,
                        (current_post_objective - visual_candidate_objective) / denominator,
                        F.cosine_similarity(
                            current_code.flatten(1), transported["visual"].flatten(1), dim=-1,
                        )[0],
                        transported["visual"].abs().mean(),
                        transported["visual"].square().mean().sqrt(),
                        transported["visual"].flatten(1, 2).std(dim=1).mean(),
                        visual_result.normalized_entropy[0],
                        visual_result.mean_max_weight[0],
                        visual_result.coverage[0],
                        current_descriptor @ source_descriptor,
                        current_descriptor.new_tensor(float(alignment.valid)),
                        current_descriptor.new_tensor(alignment.inlier_ratio),
                        current_descriptor.new_tensor(
                            alignment.normalized_median_residual if alignment.valid else 10.0
                        ),
                        current_descriptor.new_tensor(alignment.correspondences / 2048.0),
                        source_post_objective / source_pre_objective.detach().abs().clamp_min(1e-6),
                        (source_pre_objective - source_post_objective)
                        / source_pre_objective.detach().abs().clamp_min(1e-6),
                        source_pre_stats["track_coverage"],
                        source_pre_stats["mean_3d_residual"],
                        source_post_stats["mean_3d_residual"],
                        current_pre_stats["track_coverage"],
                        current_pre_stats["mean_3d_residual"],
                        current_post_stats["mean_3d_residual"],
                    ))
                    router_features.append({
                        "fold": fold["fold"], "index": index, "episode": episode,
                        "candidate": label,
                        "features": [float(value) for value in torch.cat((descriptor, scalars)).detach().cpu()],
                        "future_utility": float(visual_utility.detach()),
                        "current_objective_improvement": float(
                            ((current_post_objective - visual_candidate_objective) / denominator).detach()
                        ),
                    })
                    for condition in CONDITIONS[:-1]:
                        valid = condition not in ("geometry", "geometry_appearance") or alignment.valid
                        if valid:
                            code = (current_code + strength * transported[condition]).clamp(-1, 1)
                            loss = query_readout_loss(head, replace(current_zero, code=code), query)
                            utility = normalized_future_utility(current_query, loss)
                            loss_value, utility_value = float(loss.detach()), float(utility.detach())
                        else:
                            loss_value, utility_value = None, None
                        rows.append({
                            "fold": fold["fold"], "index": index, "episode": episode,
                            "candidate": label, "condition": condition, "valid": bool(valid),
                            "query_loss": loss_value, "utility": utility_value,
                            "alignment_inlier_ratio": alignment.inlier_ratio,
                            "alignment_residual": (
                                alignment.normalized_median_residual if alignment.valid else None
                            ),
                        })
                pooled_code = torch.stack(visual_codes).mean(dim=0)
                pooled_candidate = (current_code + strength * pooled_code).clamp(-1, 1)
                pooled_loss = query_readout_loss(
                    head, replace(current_zero, code=pooled_candidate), query,
                )
                pooled_utility = normalized_future_utility(current_query, pooled_loss)
                rows.append({
                    "fold": fold["fold"], "index": index, "episode": episode,
                    "candidate": "mean_pool_5", "condition": "visual_mean_pool", "valid": True,
                    "query_loss": float(pooled_loss.detach()),
                    "utility": float(pooled_utility.detach()),
                    "alignment_inlier_ratio": None, "alignment_residual": None,
                })
                print(json.dumps({"fold": fold["fold"], "episode": episode, "done": True}), flush=True)
        del head
    if seen != set(range(len(records))):
        raise RuntimeError("cross-fit held-out episodes do not partition train")
    summary = {condition: _summarize(rows, condition, epsilon) for condition in CONDITIONS}
    payload = {
        "experiment": "EXP-006", "stage": "stage1_crossfit_transport_ablation",
        "split": "train", "protocol_revision": config["protocol_revision"],
        "crossfit_manifest": args.crossfit, "utility_epsilon": epsilon,
        "reuse_application": config["stage1"]["reuse_application"],
        "reuse_strength": strength, "query_readout": "visual_only",
        "query_geometry_accessed": False, "validation_accessed": False,
        "mean_current_to_base": float(np.mean([row["current_to_base"] for row in current_rows])),
        "summary": summary, "episode_current": current_rows, "rows": rows,
        "router_feature_contract": {
            "query_or_future_input": False,
            "descriptor_dimensions": 256,
            "observable_scalar_dimensions": 24,
            "total_dimensions": 280,
        },
        "router_features": router_features,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({
        "out": str(output), "mean_current_to_base": payload["mean_current_to_base"],
        "summary": summary,
    }), flush=True)


if __name__ == "__main__":
    main()
