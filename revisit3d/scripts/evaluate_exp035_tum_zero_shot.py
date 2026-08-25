#!/usr/bin/env python3
"""One-shot frozen causal memory evaluation on TUM RGB-D streams."""
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
from PIL import Image

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _cpu_atom, _device_atom
from revisit3d.scripts.evaluate_exp010_absolute_geometry import _depth_metrics
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.train_exp024_metric_aligned_atom import _query_depth


PRIMARY = ("silog", "abs_rel", "point_epe_m")
METRICS = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")


def _score(compiled: dict, current: torch.Tensor, source: torch.Tensor) -> float:
    current_np = current.numpy().astype(np.float64)
    source_np = source.numpy().astype(np.float64)
    return float(
        compiled["intercept"]
        + current_np @ compiled["current"]
        + source_np @ (compiled["source"] + current_np * compiled["interaction"])
    )


def _depth_grid(frame: dict, side: int, minimum: float, maximum: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(Image.open(frame["depth"]), dtype=np.float64) / float(frame["depth_scale"])
    valid_raw = np.isfinite(raw) & (raw >= minimum) & (raw <= maximum)
    height, width = raw.shape
    y_edges = np.linspace(0, height, side + 1).round().astype(int)
    x_edges = np.linspace(0, width, side + 1).round().astype(int)
    depth = np.zeros((side, side), dtype=np.float64)
    valid = np.zeros((side, side), dtype=bool)
    for y in range(side):
        for x in range(side):
            patch = raw[y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]]
            mask = valid_raw[y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]]
            if mask.any():
                depth[y, x] = float(np.median(patch[mask]))
                valid[y, x] = True
    return depth, valid


def _query_depth_gt(event: dict, side: int, config: dict) -> tuple[np.ndarray, np.ndarray]:
    depths, masks = [], []
    for frame in event["query"]:
        augmented = {**frame, "depth_scale": event["depth_scale"]}
        depth, valid = _depth_grid(
            augmented,
            side,
            float(config["depth"]["minimum_m"]),
            float(config["depth"]["maximum_m"]),
        )
        depths.append(depth)
        masks.append(valid)
    return np.stack(depths), np.stack(masks)


def _summary(rows: list[dict], policy: str) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    return {
        "targets": len(rows),
        "sequences": len(sequences),
        **{
            metric: float(
                np.mean(
                    [
                        np.mean([row[policy][metric] for row in rows if row["sequence"] == sequence])
                        for sequence in sequences
                    ]
                )
            )
            for metric in METRICS
        },
    }


def _per_sequence(rows: list[dict], policy: str) -> dict:
    return {
        sequence: {
            "targets": len(selected),
            **{metric: float(np.mean([row[policy][metric] for row in selected])) for metric in METRICS},
        }
        for sequence in sorted({row["sequence"] for row in rows})
        for selected in [[row for row in rows if row["sequence"] == sequence]]
    }


def _bootstrap(rows: list[dict], left: str, metric: str, samples: int, seed: int) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    values = np.asarray(
        [
            np.mean(
                [row[left][metric] - row["full"][metric] for row in rows if row["sequence"] == sequence]
            )
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "sequence_improvements": {sequence: float(value) for sequence, value in zip(sequences, values)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-035_tum_zero_shot_transfer_v10.yaml")
    parser.add_argument("--confirm-zero-shot-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_zero_shot_evaluation:
        raise SystemExit("EXP-035 depth evaluation requires explicit confirmation")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-035 result already exists")
    stage1 = json.loads(Path(config["stage1"]["result"]).read_text())
    cache_path = Path(config["stage1"]["cache"])
    atom_path = Path(config["model"]["atom_checkpoint"])
    address_path = Path(config["model"]["address_artifact"])
    if not (
        stage1["rows"] == 223
        and stage1["image_decoded"]
        and stage1["model_output_accessed"]
        and not stage1["depth_decoded"]
        and not stage1["tum_fit_performed"]
        and stage1["cache_sha256"] == _sha256(cache_path)
        and _sha256(atom_path) == config["model"]["atom_sha256"]
        and _sha256(address_path) == config["model"]["address_sha256"]
    ):
        raise RuntimeError("EXP-035 frozen contract failed")

    events = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    atom_checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    address = joblib.load(address_path)
    device = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    compiled = address["compiled_mips"]
    threshold = float(address["acceptance_threshold"])
    capacity = int(config["bank"]["capacity"])
    strength = float(config["method"]["reuse_strength"])
    rows = []

    with torch.enable_grad():
        for sequence in sorted({event["sequence"] for event in events}):
            indices = [index for index, event in enumerate(events) if event["sequence"] == sequence]
            indices.sort(key=lambda index: (events[index]["timestamp"], events[index]["event_id"]))
            memory = {}
            bank = []
            seen = 0
            rng = random.Random(
                int(config["seed"]) + int(hashlib.sha1(sequence.encode()).hexdigest()[:8], 16)
            )
            for index in indices:
                event = events[index]
                current = CachedAtomSegment.from_cache(geometry["rows"][index]["segments"]["context"], "current", device)
                zero = current.atom(head)
                code = adapt_minimal(
                    head, current, zero.code, step_size=float(config["method"]["step_size"])
                )
                current_atom = replace(zero, code=code)
                state = {
                    "atom": _cpu_atom(replace(zero, code=code.detach())),
                    "descriptor": zero.key.mean((1, 2))[0].detach().cpu(),
                }

                if event["is_revisit_target"] and bank:
                    query = CachedAtomSegment.from_cache(geometry["rows"][index]["segments"]["query"], "query", device)
                    query_zero = query.atom(head)
                    current_prediction = _query_depth(head, query, query_zero, current_atom)
                    scores = {key: _score(compiled, state["descriptor"], memory[key]["descriptor"]) for key in bank}
                    winner = max(bank, key=lambda key: (scores[key], key))
                    accepted = scores[winner] > threshold
                    candidate_predictions = {}
                    for source_key in bank:
                        source = _device_atom(memory[source_key]["atom"], device)
                        transported = visual_transport(source, zero).code
                        candidate = replace(zero, code=(code + strength * transported).clamp(-1, 1))
                        candidate_predictions[source_key] = _query_depth(head, query, query_zero, candidate)

                    # Offline dense depth is decoded only after every online decision/prediction.
                    side = current_prediction.shape[-1]
                    target_depth, target_valid = _query_depth_gt(event, side, config)
                    intrinsics = query.intrinsics[0].detach().cpu().numpy()

                    def metrics(prediction: torch.Tensor):
                        return _depth_metrics(
                            prediction.detach().cpu().numpy(),
                            target_depth,
                            target_valid,
                            intrinsics,
                            image_size=query.image_size,
                            minimum_cells=int(config["depth"]["minimum_cells_per_view"]),
                        )

                    current_metrics = metrics(current_prediction)
                    candidate_metrics = {key: metrics(value) for key, value in candidate_predictions.items()}
                    if current_metrics is not None and all(value is not None for value in candidate_metrics.values()):
                        appearance = max(
                            bank,
                            key=lambda key: float(
                                state["descriptor"] @ memory[key]["descriptor"]
                                / max(state["descriptor"].norm() * memory[key]["descriptor"].norm(), 1e-12)
                            ),
                        )
                        full = candidate_metrics[winner] if accepted else current_metrics
                        appearance_metrics = candidate_metrics[appearance] if accepted else current_metrics
                        random_metrics = {
                            metric: float(np.mean([candidate_metrics[key][metric] for key in bank]))
                            if accepted
                            else current_metrics[metric]
                            for metric in METRICS
                        }
                        oracle = min(candidate_metrics.values(), key=lambda value: value["silog"])
                        rows.append(
                            {
                                "target": event["event_id"],
                                "sequence": sequence,
                                "accepted": accepted,
                                "bank_size": len(bank),
                                "current": current_metrics,
                                "full": full,
                                "random": random_metrics,
                                "appearance": appearance_metrics,
                                "oracle": oracle,
                            }
                        )

                seen += 1
                key = event["event_id"]
                if len(bank) < capacity:
                    bank.append(key)
                    memory[key] = state
                else:
                    replacement = rng.randrange(seen)
                    if replacement < capacity:
                        old = bank[replacement]
                        del memory[old]
                        bank[replacement] = key
                        memory[key] = state
            print(json.dumps({"sequence": sequence, "evaluated": sum(row["sequence"] == sequence for row in rows)}), flush=True)

    policies = ("current", "full", "random", "appearance", "oracle")
    summaries = {policy: _summary(rows, policy) for policy in policies}
    per_sequence = {policy: _per_sequence(rows, policy) for policy in policies}
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    comparisons = {
        f"full_vs_{left}": {
            metric: _bootstrap(rows, left, metric, samples, seed + offset + metric_index)
            for metric_index, metric in enumerate(PRIMARY)
        }
        for left, offset in (("current", 0), ("random", 10), ("appearance", 20))
    }
    checks = {
        "coverage": len(rows) >= int(config["success"]["minimum_targets"])
        and len({row["sequence"] for row in rows}) >= int(config["success"]["minimum_sequences"])
    }
    for control in ("current", "random", "appearance"):
        checks[f"full_better_{control}_all_primary_means"] = all(
            summaries["full"][metric] < summaries[control][metric] for metric in PRIMARY
        )
    result = {
        "experiment": "EXP-035",
        "stage": "tum_frozen_zero_shot_evaluation",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "split": "tum_zero_shot",
        "tum_fit_performed": False,
        "terminal_no_tum_repair": True,
        "artifact_hashes": {"atom": _sha256(atom_path), "address": _sha256(address_path)},
        "cache_sha256": _sha256(cache_path),
        "acceptance": float(np.mean([row["accepted"] for row in rows])),
        "summaries": summaries,
        "per_sequence": per_sequence,
        "comparisons": comparisons,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"summaries": summaries, "comparisons": comparisons, "gate": result["registered_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
