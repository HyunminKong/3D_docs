#!/usr/bin/env python3
"""Build fold-local, leakage-safe causal utility tables for EXP-007 v1.2."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    CachedAtomSegment,
    adapt_context,
    geometry_objective,
    observable_router_features,
    primary_feature_columns,
    require_exp006_split,
)
from revisit3d.losses import normalized_future_utility
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.build_exp007_all_memory_table import (
    MemoryState,
    _dummy_alignment,
    _identifier,
    _identity,
    _query_loss,
    _sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-007_causal_bank_oof_v12.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-007 OOF utility table requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("protocol_revision") != "v1.2":
        raise RuntimeError("expected registered EXP-007 v1.2 config")
    require_exp006_split(config["data"]["split"])
    output = Path(config["stage0"]["utility_table"])
    if output.exists():
        raise RuntimeError(f"EXP-007 v1.2 table already exists: {output}")

    cache_path = Path(config["atom"]["cache"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    crossfit_path = Path(config["atom"]["crossfit"])
    crossfit = json.loads(crossfit_path.read_text())
    router_source_path = Path(config["router"]["train_features"])
    router_source = json.loads(router_source_path.read_text())
    expected_revision = config["atom"]["source_protocol_revision"]
    if not (
        cache.get("split") == crossfit.get("split") == router_source.get("split") == "train"
        and cache.get("protocol_revision") == crossfit.get("protocol_revision")
        == router_source.get("protocol_revision") == expected_revision
        and crossfit.get("validation_accessed") is False
        and router_source.get("validation_accessed") is False
        and router_source.get("query_geometry_accessed") is False
    ):
        raise RuntimeError("OOF cache/head/router sources violate the train-only contract")

    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    records = dataset.records
    if len(records) != 76 or len(cache["rows"]) != len(records):
        raise RuntimeError("OOF table expects 76 expanded-train episodes")
    covered = sorted(index for fold in crossfit["folds"] for index in fold["held_out"])
    if covered != list(range(len(records))):
        raise RuntimeError("cross-fit folds do not partition train episodes")

    router_rows = router_source["router_features"]
    router_matrix = np.asarray([row["features"] for row in router_rows], dtype=np.float64)
    router_targets = np.asarray([row["future_utility"] for row in router_rows], dtype=np.float64)
    router_folds = np.asarray([row["fold"] for row in router_rows], dtype=np.int64)
    columns = primary_feature_columns()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    atom_config = config["atom"]
    streams = []
    total_pairs = 0
    started = time.perf_counter()
    tag_descriptor = {"a_context": "a", "b_context": "b", "a_prime_context": "a_prime"}

    with torch.enable_grad():
        for fold_spec in crossfit["folds"]:
            fold_index = int(fold_spec["fold"])
            held_out = [int(index) for index in fold_spec["held_out"]]
            checkpoint_path = Path(fold_spec["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not (
                checkpoint.get("protocol_revision") == expected_revision
                and checkpoint.get("held_out") == held_out
                and checkpoint.get("query_readout") == "visual_only"
            ):
                raise RuntimeError(f"fold-{fold_index} atom checkpoint mismatch")
            head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
            head.load_state_dict(checkpoint["head"])
            head.eval().requires_grad_(False)

            train_router = router_folds != fold_index
            router = make_pipeline(
                StandardScaler(),
                PCA(
                    n_components=int(config["router"]["pca_components"]),
                    random_state=int(config["router"]["pca_random_state"]),
                ),
                Ridge(alpha=float(config["router"]["ridge_alpha"])),
            )
            router.fit(router_matrix[train_router][:, columns], router_targets[train_router])

            unique = {}
            events = []
            for index in held_out:
                record = records[index]
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
                    unique[identity]["observations"].append({
                        "episode_index": index, "tag": cache_tag,
                    })
                    identifiers[cache_tag] = identifier
                events.append({
                    "episode_index": index,
                    "episode": record["episode_id"],
                    "pre_query_writes": [identifiers["a_context"], identifiers["b_context"]],
                    "post_query_writes": [identifiers["a_prime_context"]],
                })
            contexts = sorted(unique.values(), key=lambda row: row["context_id"])
            memory_states = {}
            for context in contexts:
                payload = cache["rows"][context["cache_index"]]["segments"][context["cache_tag"]]
                segment = CachedAtomSegment.from_cache(payload, "source", device)
                zero = segment.atom(head)
                code, _ = adapt_context(
                    head, segment, zero.code,
                    step_size=float(atom_config["ttt_step_size"]),
                    steps=int(atom_config["ttt_steps"]),
                )
                pre, pre_stats = geometry_objective(head, segment, zero.code, return_stats=True)
                post, post_stats = geometry_objective(head, segment, code, return_stats=True)
                atom = replace(zero, code=code.detach()).detach()
                memory_states[context["context_id"]] = MemoryState(
                    atom=atom,
                    descriptor=atom.key.mean(dim=(1, 2))[0].detach(),
                    pre_objective=pre.detach(),
                    post_objective=post.detach(),
                    pre_stats={key: value.detach() for key, value in pre_stats.items()},
                    post_stats={key: value.detach() for key, value in post_stats.items()},
                )

            pair_rows = []
            current_rows = []
            dummy = None
            for event_position, index in enumerate(held_out):
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
                features = []
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
                    vector = observable_router_features(
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
                    features.append(vector.detach().cpu().numpy()[columns])
                    pair_rows.append({
                        "fold": fold_index,
                        "episode": records[index]["episode_id"],
                        "episode_index": index,
                        "context_id": context["context_id"],
                        "future_utility": float(utility.detach()),
                        "appearance_similarity": float(vector[267].detach()),
                        "current_objective_improvement": float(vector[259].detach()),
                        "predicted_utility": None,
                    })
                prediction = router.predict(np.asarray(features, dtype=np.float64))
                for row, value in zip(pair_rows[offset:], prediction):
                    row["predicted_utility"] = float(value)
                current_rows.append({
                    "episode": records[index]["episode_id"],
                    "episode_index": index,
                    "base_query": float(base_query.detach()),
                    "current_query": float(current_query.detach()),
                    "current_to_base": float(
                        (current_query / base_query.detach().abs().clamp_min(1e-6)).detach()
                    ),
                    "wall_time_s": time.perf_counter() - episode_started,
                })
                print(json.dumps({
                    "fold": fold_index, "completed": event_position + 1, "total": len(held_out),
                    "episode": records[index]["episode_id"],
                }), flush=True)

            streams.append({
                "fold": fold_index,
                "held_out": held_out,
                "atom_checkpoint": str(checkpoint_path),
                "atom_checkpoint_sha256": _sha256(checkpoint_path),
                "router_train_candidates": int(train_router.sum()),
                "contexts": [{
                    **{key: value for key, value in context.items() if key not in ("cache_index", "cache_tag")},
                    "descriptor": [
                        float(value)
                        for value in memory_states[context["context_id"]].descriptor.cpu()
                    ],
                } for context in contexts],
                "events": events,
                "current": current_rows,
                "pairs": pair_rows,
            })
            total_pairs += len(pair_rows)
            del head, memory_states
            torch.cuda.empty_cache()

    result = {
        "experiment": "EXP-007",
        "stage": "stage0_oof_fold_local_utility_table",
        "split": "train",
        "protocol_revision": "v1.2",
        "validation_accessed": False,
        "test_accessed": False,
        "query_or_future_router_input": False,
        "query_role": "offline_utility_label_only",
        "fold_local_memory_coordinates": True,
        "cross_fold_bank_mixing": False,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "crossfit": str(crossfit_path),
        "router_train_features": str(router_source_path),
        "folds": len(streams),
        "episodes": sum(len(stream["events"]) for stream in streams),
        "pair_evaluations": total_pairs,
        "utility_deadband": float(atom_config["utility_deadband"]),
        "reuse_strength": float(atom_config["reuse_strength"]),
        "runtime": {
            "wall_time_s": time.perf_counter() - started,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        },
        "streams": streams,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(output), "folds": len(streams), "episodes": result["episodes"],
        "pairs": total_pairs, "runtime": result["runtime"],
    }), flush=True)


if __name__ == "__main__":
    main()
