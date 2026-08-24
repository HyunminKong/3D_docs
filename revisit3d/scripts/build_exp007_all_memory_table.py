#!/usr/bin/env python3
"""Build the exhaustive train-only utility table for EXP-007 Stage 0.

The table is deliberately exhaustive and offline. Future query observations
produce labels only; causal availability is applied later by the bank
simulator and is never a router input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    CachedAtomSegment,
    adapt_context,
    geometry_objective,
    observable_router_features,
    primary_feature_columns,
    query_readout_loss,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import PlasticityAtom, Sim3Alignment, SpatialPlasticityHead, visual_transport


@dataclass
class MemoryState:
    atom: PlasticityAtom
    descriptor: torch.Tensor
    pre_objective: torch.Tensor
    post_objective: torch.Tensor
    pre_stats: dict[str, torch.Tensor]
    post_stats: dict[str, torch.Tensor]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(descriptor: dict) -> tuple[str, tuple[int, ...]]:
    return descriptor["scene"], tuple(int(value) for value in descriptor["frames"])


def _identifier(identity: tuple[str, tuple[int, ...]]) -> str:
    scene, frames = identity
    token = f"{scene}:{','.join(map(str, frames))}"
    return f"{scene}:{hashlib.sha256(token.encode()).hexdigest()[:12]}"


def _dummy_alignment(reference: torch.Tensor) -> Sim3Alignment:
    return Sim3Alignment(
        scale=reference.new_ones(()),
        rotation=torch.eye(3, device=reference.device, dtype=reference.dtype),
        translation=reference.new_zeros(3),
        valid=False,
        correspondences=0,
        inliers=0,
        inlier_ratio=0.0,
        normalized_median_residual=float("inf"),
        source_rank_ratio=0.0,
        target_rank_ratio=0.0,
    )


def _query_loss(
    head: SpatialPlasticityHead,
    current_zero: PlasticityAtom,
    current_code: torch.Tensor,
    query: CachedAtomSegment,
    query_zero: PlasticityAtom,
) -> torch.Tensor:
    context_atom = replace(current_zero, code=current_code)
    query_code = visual_transport(context_atom, query_zero).code
    return geometry_objective(head, query, query_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_v10.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-007 all-memory table requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("experiment") != "EXP-007" or config.get("protocol_revision") != "v1.0":
        raise RuntimeError("expected the registered EXP-007 v1.0 protocol")
    require_exp006_split(config["data"]["split"])
    output = Path(config["stage0"]["utility_table"])
    if output.exists():
        raise RuntimeError(f"EXP-007 Stage-0 table already exists: {output}")

    atom_config = config["atom"]
    cache_path = Path(atom_config["cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not (
        cache.get("split") == "train"
        and cache.get("protocol_revision") == atom_config["source_protocol_revision"]
    ):
        raise RuntimeError("EXP-007 requires the expanded train-only v2.7 cache")
    atom_path = Path(atom_config["checkpoint"])
    atom_checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    if not (
        atom_checkpoint.get("split") == "train"
        and atom_checkpoint.get("protocol_revision") == atom_config["source_protocol_revision"]
        and atom_checkpoint.get("query_readout") == "visual_only"
    ):
        raise RuntimeError("EXP-007 atom checkpoint contract mismatch")
    router_path = Path(config["router"]["model"])
    router_fit_path = Path(config["router"]["fit_result"])
    router_fit = json.loads(router_fit_path.read_text())
    if not (
        router_fit.get("validation_accessed") is False
        and router_fit.get("model_sha256") == _sha256(router_path)
    ):
        raise RuntimeError("EXP-007 router must be the frozen train-only D023 artifact")
    router_payload = joblib.load(router_path)
    columns = primary_feature_columns()
    if router_payload.get("feature_columns") != columns:
        raise RuntimeError("EXP-007 router feature contract mismatch")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    records = dataset.records
    if len(records) != 76 or len(cache["rows"]) != len(records):
        raise RuntimeError("EXP-007 v1.0 expects the 76-episode expanded train split")

    tag_descriptor = {"a_context": "a", "b_context": "b", "a_prime_context": "a_prime"}
    unique: dict[tuple[str, tuple[int, ...]], dict] = {}
    event_writes = []
    for index, record in enumerate(records):
        identifiers = {}
        for cache_tag, record_tag in tag_descriptor.items():
            identity = _identity(record[record_tag])
            identifier = _identifier(identity)
            if identity not in unique:
                unique[identity] = {
                    "context_id": identifier,
                    "scene": identity[0],
                    "frames": list(identity[1]),
                    "cache_index": index,
                    "cache_tag": cache_tag,
                    "observations": [],
                }
            elif unique[identity]["context_id"] != identifier:
                raise RuntimeError("context identifier collision")
            unique[identity]["observations"].append({"episode_index": index, "tag": cache_tag})
            identifiers[cache_tag] = identifier
        event_writes.append({
            "index": index,
            "episode": record["episode_id"],
            "pre_query_writes": [identifiers["a_context"], identifiers["b_context"]],
            "post_query_writes": [identifiers["a_prime_context"]],
        })
    contexts = sorted(unique.values(), key=lambda row: row["context_id"])
    if len(contexts) != 123:
        raise RuntimeError(f"registered context audit expected 123 unique contexts, got {len(contexts)}")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    memory_states: dict[str, MemoryState] = {}
    started = time.perf_counter()
    with torch.enable_grad():
        for position, context in enumerate(contexts):
            payload = cache["rows"][context["cache_index"]]["segments"][context["cache_tag"]]
            segment = CachedAtomSegment.from_cache(payload, "source", device)
            zero = segment.atom(head)
            code, _ = adapt_context(
                head, segment, zero.code,
                step_size=float(atom_config["ttt_step_size"]),
                steps=int(atom_config["ttt_steps"]),
            )
            pre_objective, pre_stats = geometry_objective(
                head, segment, zero.code, return_stats=True,
            )
            post_objective, post_stats = geometry_objective(
                head, segment, code, return_stats=True,
            )
            atom = replace(zero, code=code.detach()).detach()
            memory_states[context["context_id"]] = MemoryState(
                atom=atom,
                descriptor=atom.key.mean(dim=(1, 2))[0].detach(),
                pre_objective=pre_objective.detach(),
                post_objective=post_objective.detach(),
                pre_stats={key: value.detach() for key, value in pre_stats.items()},
                post_stats={key: value.detach() for key, value in post_stats.items()},
            )
            if position == 0 or (position + 1) % 20 == 0 or position + 1 == len(contexts):
                print(json.dumps({
                    "phase": "memory_atoms", "completed": position + 1, "total": len(contexts),
                }), flush=True)

        pair_rows = []
        episode_rows = []
        dummy = None
        for index, record in enumerate(records):
            episode_started = time.perf_counter()
            cached_row = cache["rows"][index]
            current = CachedAtomSegment.from_cache(
                cached_row["segments"]["a_prime_context"], "current", device,
            )
            query = CachedAtomSegment.from_cache(
                cached_row["segments"]["a_prime_query"], "query", device,
            )
            current_zero = current.atom(head)
            query_zero = query.atom(head)
            base_query = _query_loss(head, current_zero, current_zero.code, query, query_zero)
            current_code, _ = adapt_context(
                head, current, current_zero.code,
                step_size=float(atom_config["ttt_step_size"]),
                steps=int(atom_config["ttt_steps"]),
            )
            current_query = _query_loss(head, current_zero, current_code, query, query_zero)
            current_pre, current_pre_stats = geometry_objective(
                head, current, current_zero.code, return_stats=True,
            )
            current_post, current_post_stats = geometry_objective(
                head, current, current_code, return_stats=True,
            )
            current_descriptor = current_zero.key.mean(dim=(1, 2))[0]
            if dummy is None:
                dummy = _dummy_alignment(current_descriptor)
            feature_rows = []
            offset = len(pair_rows)
            for context in contexts:
                memory = memory_states[context["context_id"]]
                visual = visual_transport(memory.atom, current_zero)
                candidate_code = (
                    current_code + float(atom_config["reuse_strength"]) * visual.code
                ).clamp(-1, 1)
                candidate_objective = geometry_objective(head, current, candidate_code)
                candidate_query = _query_loss(
                    head, current_zero, candidate_code, query, query_zero,
                )
                utility = normalized_future_utility(current_query, candidate_query)
                features = observable_router_features(
                    current_descriptor=current_descriptor,
                    source_descriptor=memory.descriptor,
                    current_code=current_code,
                    transported_code=visual.code,
                    visual_result=visual,
                    alignment=dummy,
                    current_pre_objective=current_pre,
                    current_post_objective=current_post,
                    candidate_objective=candidate_objective,
                    source_pre_objective=memory.pre_objective,
                    source_post_objective=memory.post_objective,
                    current_pre_stats=current_pre_stats,
                    current_post_stats=current_post_stats,
                    source_pre_stats=memory.pre_stats,
                    source_post_stats=memory.post_stats,
                )
                feature_rows.append(features.detach().cpu().numpy()[columns])
                pair_rows.append({
                    "episode": record["episode_id"],
                    "episode_index": index,
                    "context_id": context["context_id"],
                    "future_utility": float(utility.detach()),
                    "appearance_similarity": float(features[267].detach()),
                    "current_objective_improvement": float(features[259].detach()),
                    "predicted_utility": None,
                })
            prediction = router_payload["model"].predict(np.asarray(feature_rows, dtype=np.float64))
            for row, value in zip(pair_rows[offset:], prediction):
                row["predicted_utility"] = float(value)
            episode_rows.append({
                "episode": record["episode_id"],
                "episode_index": index,
                "base_query": float(base_query.detach()),
                "current_query": float(current_query.detach()),
                "current_to_base": float(
                    (current_query / base_query.detach().abs().clamp_min(1e-6)).detach()
                ),
                "candidate_count": len(contexts),
                "wall_time_s": time.perf_counter() - episode_started,
            })
            print(json.dumps({
                "phase": "utility_table", "completed": index + 1, "total": len(records),
                "episode": record["episode_id"], "wall_time_s": episode_rows[-1]["wall_time_s"],
            }), flush=True)

    context_rows = [{
        **{key: value for key, value in context.items() if key not in ("cache_index", "cache_tag")},
        "descriptor": [
            float(value) for value in memory_states[context["context_id"]].descriptor.cpu()
        ],
    } for context in contexts]
    result = {
        "experiment": "EXP-007",
        "stage": "stage0_all_memory_utility_table",
        "split": "train",
        "protocol_revision": config["protocol_revision"],
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "offline_utility_label_only",
        "causal_mask_applied_during_simulation": True,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "atom_checkpoint": str(atom_path),
        "atom_checkpoint_sha256": _sha256(atom_path),
        "router_model": str(router_path),
        "router_model_sha256": _sha256(router_path),
        "episodes": len(records),
        "unique_contexts": len(contexts),
        "context_observations": sum(len(row["observations"]) for row in contexts),
        "pair_evaluations": len(pair_rows),
        "utility_deadband": float(atom_config["utility_deadband"]),
        "reuse_strength": float(atom_config["reuse_strength"]),
        "current": episode_rows,
        "contexts": context_rows,
        "events": event_writes,
        "pairs": pair_rows,
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "mean_episode_table_s": float(np.mean([row["wall_time_s"] for row in episode_rows])),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "episodes": len(records), "contexts": len(contexts),
        "pairs": len(pair_rows), "runtime": result["runtime"],
    }), flush=True)


if __name__ == "__main__":
    main()
