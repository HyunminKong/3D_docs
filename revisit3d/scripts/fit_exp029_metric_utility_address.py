#!/usr/bin/env python3
"""Fit the single deployable address to sparse-LiDAR log-depth utility."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import torch

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.models import visual_transport
from revisit3d.scripts import fit_exp016_unified_utility_address as protocol
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _cpu_atom, _device_atom, _identifier, _timestamp,
)
from revisit3d.scripts.train_exp024_metric_aligned_atom import (
    _lidar_cache, _metric_loss, _query_depth,
)


def _build_metric_pairs(config, head, geometry, manifest, device):
    context, targets, locations = protocol._context_tables(manifest)
    metadata_cache = {}
    scene_root = Path(config["data"]["scene_root"])
    for info in context.values():
        info["timestamp"] = _timestamp(info["segment"], scene_root, metadata_cache)
    events = sorted(context.values(), key=lambda row: (row["timestamp"], row["id"]))
    lidar = _lidar_cache(manifest, geometry, config, device)
    memory, bank, features, utility, metadata = {}, [], [], [], []
    target_table = {
        target["episode"]: {"component": target["component"], "location": target["location"]}
        for target in targets.values()
    }
    panel_size = int(config["method"]["panel_size"])
    step_size = float(config["method"]["step_size"])
    strength = float(config["method"]["reuse_strength"])
    with torch.enable_grad():
        for event_index, event in enumerate(events):
            key = event["id"]
            payload = geometry["rows"][event["cache_index"]]["segments"][event["cache_tag"]]
            role = "current" if key in targets else "source"
            segment = CachedAtomSegment.from_cache(payload, role, device)
            zero = segment.atom(head)
            code = adapt_minimal(head, segment, zero.code, step_size=step_size)
            atom = replace(zero, code=code.detach())
            state = {
                "atom": _cpu_atom(atom),
                "descriptor": zero.key.mean(dim=(1, 2))[0].detach().cpu(),
                "location": event["location"],
            }
            if key in targets and bank:
                target = targets[key]
                stable = int(hashlib.sha1(target["episode"].encode()).hexdigest()[:8], 16)
                generator = random.Random(int(config["seed"]) + stable)
                panel = generator.sample(bank, min(panel_size, len(bank)))
                query_payload = geometry["rows"][target["cache_index"]]["segments"]["a_prime_query"]
                query = CachedAtomSegment.from_cache(query_payload, "query", device)
                query_zero = query.atom(head)
                current_loss = _metric_loss(
                    _query_depth(head, query, query_zero, atom), *lidar[target["cache_index"]], config
                )
                for candidate in panel:
                    source_state = memory[candidate]
                    source_atom = _device_atom(source_state["atom"], device)
                    transported = visual_transport(source_atom, zero).code
                    candidate_atom = replace(
                        zero, code=(code + strength * transported).clamp(-1, 1)
                    )
                    candidate_loss = _metric_loss(
                        _query_depth(head, query, query_zero, candidate_atom),
                        *lidar[target["cache_index"]], config,
                    )
                    value = ((current_loss - candidate_loss) / current_loss.detach().abs().clamp_min(1e-6)).detach()
                    features.append(protocol._pair_features(state["descriptor"], source_state["descriptor"]))
                    utility.append(value.cpu())
                    metadata.append({
                        "episode": target["episode"], "component": target["component"],
                        "target_context": key, "source_context": candidate,
                        "target_location": target["location"],
                        "source_location": source_state["location"],
                    })
            memory[key] = state
            bank.append(key)
            if (event_index + 1) % 50 == 0 or event_index + 1 == len(events):
                print(json.dumps({
                    "events": event_index + 1, "total": len(events),
                    "targets": len({row['episode'] for row in metadata}), "pairs": len(metadata),
                }), flush=True)
    matrix = torch.stack(features).float()
    target = torch.stack(utility).float()
    if matrix.shape != (len(metadata), 256) or not torch.isfinite(matrix).all() or not torch.isfinite(target).all():
        raise RuntimeError("EXP-029 metric pair tensor contract failed")
    return matrix, target, metadata, target_table


if __name__ == "__main__":
    protocol._build_pairs = _build_metric_pairs
    protocol.main()
