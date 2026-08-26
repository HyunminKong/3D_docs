#!/usr/bin/env python3
"""Absolute-geometry audit of the frozen current-only exact-meta coordinate."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from revisit3d.backbones import FrozenCUT3RCarrier, LocalTokenResidual
from revisit3d.scripts.evaluate_exp010_absolute_geometry import _depth_metrics
from revisit3d.scripts.evaluate_exp035_tum_zero_shot import _query_depth_gt
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import (
    METRICS,
    _build_sequence,
    _views,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp042_learned_cut3r_plasticity_coordinate import (
    _online_code,
    _points,
)


PRIMARY = ("silog", "abs_rel", "point_epe_m")
POLICIES = ("cut3r", "generic_one", "exact_one", "exact_two")


def _next_code(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    code: torch.Tensor,
    previous_points: torch.Tensor,
    *,
    normalized_step: float,
    patch_size: int,
) -> torch.Tensor:
    current = code.detach().clone().requires_grad_(True)
    prediction = carrier.readout(auxiliary, code=current)
    points = _points(prediction, patch_size)
    distances = torch.linalg.vector_norm(
        points[:, :, None, :] - previous_points[:, None, :, :], dim=-1
    )
    loss = 0.5 * (
        distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean()
    )
    gradient = torch.autograd.grad(loss, current, create_graph=False)[0]
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    return current.detach() - float(normalized_step) * normalized.detach()


def _depth(prediction: dict, side: int) -> tuple[np.ndarray, tuple[int, int]]:
    dense = prediction["pts3d_in_self_view"][0, ..., 2].float()
    height, width = dense.shape
    pooled = F.interpolate(dense[None, None], size=(side, side), mode="area")[0, 0]
    return pooled.detach().cpu().numpy(), (height, width)


def _summary(rows: list[dict], policy: str) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    return {
        "targets": len(rows),
        "sequences": len(sequences),
        **{
            metric: float(
                np.mean(
                    [
                        np.mean(
                            [
                                row[policy][metric]
                                for row in rows
                                if row["sequence"] == sequence
                            ]
                        )
                        for sequence in sequences
                    ]
                )
            )
            for metric in METRICS
        },
    }


def _bootstrap(
    rows: list[dict], comparisons: dict[str, tuple[str, str]], config: dict
) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    draws = int(config["analysis"]["bootstrap_draws"])
    seed = int(config["analysis"]["bootstrap_seed"])
    result = {}
    for comparison_index, (name, (left, right)) in enumerate(comparisons.items()):
        result[name] = {}
        for metric_index, metric in enumerate(PRIMARY):
            values = np.asarray(
                [
                    np.mean(
                        [
                            row[left][metric] - row[right][metric]
                            for row in rows
                            if row["sequence"] == sequence
                        ]
                    )
                    for sequence in sequences
                ],
                dtype=np.float64,
            )
            generator = np.random.default_rng(
                seed + comparison_index * 10 + metric_index
            )
            indices = generator.integers(
                0, len(values), size=(draws, len(values))
            )
            samples = values[indices].mean(axis=1)
            result[name][metric] = {
                "direction": f"{left}_error_minus_{right}_error",
                "mean": float(values.mean()),
                "ci95": [
                    float(value) for value in np.quantile(samples, [0.025, 0.975])
                ],
                "positive_sequences": int((values > 0).sum()),
                "sequences": len(values),
                "per_sequence": {
                    sequence: float(value)
                    for sequence, value in zip(sequences, values)
                },
            }
    return result


def _copy_weight(carrier: FrozenCUT3RCarrier, weight: torch.Tensor) -> None:
    with torch.no_grad():
        carrier.residual.projection.weight.copy_(
            weight.to(
                device=carrier.residual.projection.weight.device,
                dtype=carrier.residual.projection.weight.dtype,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-050_current_only_exact_meta_tum_v10.yaml"
    )
    parser.add_argument("--confirm-exposed-tum-development", action="store_true")
    parser.add_argument("--smoke-targets", type=int, default=0)
    args = parser.parse_args()
    if not args.confirm_exposed_tum_development:
        raise SystemExit("EXP-050 requires explicit exposed-TUM confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-050 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists() and args.smoke_targets == 0:
        raise RuntimeError("EXP-050 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    baseline_path = Path(config["data"]["baseline_result"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    exact_checkpoint = Path(config["plasticity"]["exact_meta_checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(baseline_path) == config["data"]["baseline_result_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(exact_checkpoint)
        == config["plasticity"]["exact_meta_checkpoint_sha256"]
        and config["data"]["role"] == "exposed_transfer_development"
        and config["data"]["terminal_access"] is False
        and config["carrier"]["mode"] == "cut3r"
        and config["plasticity"]["persistent_state_update"] is False
        and int(config["plasticity"]["steps_primary"]) == 1
        and int(config["plasticity"]["steps_diagnostic"]) == 2
    ):
        raise RuntimeError("EXP-050 frozen contract failed")

    events = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    targets = [event for event in events if event["is_revisit_target"]]
    sequences = sorted({event["sequence"] for event in events})
    total_frames = sum(len(_build_sequence(events, scene)[0]) for scene in sequences)
    if not (
        len(targets) == int(config["data"]["exact_targets"])
        and len(sequences) == int(config["data"]["exact_sequences"])
        and total_frames == int(config["data"]["exact_stream_frames"])
        and baseline["query_update"] is False
        and baseline["sequence_reset_only"] is True
    ):
        raise RuntimeError("EXP-050 causal coverage contract failed")
    if args.smoke_targets:
        permitted = {
            event["event_id"] for event in targets[: int(args.smoke_targets)]
        }
    else:
        permitted = {event["event_id"] for event in targets}
    run_sequences = (
        sorted(
            {
                event["sequence"]
                for event in targets
                if event["event_id"] in permitted
            }
        )
        if args.smoke_targets
        else sequences
    )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    carrier = FrozenCUT3RCarrier(
        carrier_checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["generic_basis_seed"]),
    ).cuda()
    carrier.eval()
    carrier.residual.requires_grad_(False)
    generic = LocalTokenResidual(
        code_dim=int(config["plasticity"]["code_dim"]),
        token_dim=carrier.residual.token_dim,
        seed=int(config["plasticity"]["generic_basis_seed"]),
    )
    generic_weight = generic.projection.weight.detach().clone()
    payload = torch.load(exact_checkpoint, map_location="cpu")
    exact_weight = payload["residual_state_dict"]["projection.weight"].detach().clone()
    del generic, payload

    side = int(config["depth"]["grid_side"])
    patch_size = int(config["plasticity"]["patch_size"])
    step = float(config["plasticity"]["normalized_step"])
    event_by_id = {event["event_id"]: event for event in events}
    predicted: dict[str, dict[str, list[np.ndarray | None]]] = {
        event_id: {policy: [None, None, None, None] for policy in POLICIES}
        for event_id in permitted
    }
    output_shapes: dict[str, tuple[int, int]] = {}
    processed_frames = 0
    adapted_query_frames = 0
    parity_max = 0.0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for sequence_number, sequence in enumerate(run_sequences):
        paths, updates, query_positions = _build_sequence(events, sequence)
        position_to_query = {
            position: (event_id, view_index)
            for event_id, positions in query_positions.items()
            if event_id in permitted
            for view_index, position in enumerate(positions)
        }
        images = load_images_for_eval(
            paths,
            size=int(config["carrier"]["image_size"]),
            verbose=False,
            crop=bool(config["carrier"]["crop"]),
        )
        views = _views(images, updates)
        state = None
        previous_points = None
        for position, view in enumerate(views):
            with torch.no_grad():
                base_prediction, next_state, auxiliary = carrier.step(view, state)
                base_points = _points(base_prediction, patch_size)
            processed_frames += 1
            query = position_to_query.get(position)
            if query is not None:
                if previous_points is None:
                    raise RuntimeError("EXP-050 query has no causal previous frame")
                event_id, view_index = query
                base_depth, output_shape = _depth(base_prediction, side)
                predicted[event_id]["cut3r"][view_index] = base_depth
                output_shapes[event_id] = output_shape

                _copy_weight(carrier, generic_weight)
                generic_code = _online_code(
                    carrier,
                    auxiliary,
                    previous_points,
                    patch_size=patch_size,
                    normalized_step=step,
                )["code"]
                with torch.no_grad():
                    generic_prediction = carrier.readout(auxiliary, code=generic_code)
                    generic_depth, generic_shape = _depth(generic_prediction, side)
                predicted[event_id]["generic_one"][view_index] = generic_depth

                _copy_weight(carrier, exact_weight)
                exact = _online_code(
                    carrier,
                    auxiliary,
                    previous_points,
                    patch_size=patch_size,
                    normalized_step=step,
                )
                with torch.no_grad():
                    exact_prediction = carrier.readout(auxiliary, code=exact["code"])
                    exact_depth, exact_shape = _depth(exact_prediction, side)
                exact_two_code = _next_code(
                    carrier,
                    auxiliary,
                    exact["code"],
                    previous_points,
                    normalized_step=step,
                    patch_size=patch_size,
                )
                with torch.no_grad():
                    exact_two_prediction = carrier.readout(
                        auxiliary, code=exact_two_code
                    )
                    exact_two_depth, exact_two_shape = _depth(
                        exact_two_prediction, side
                    )
                predicted[event_id]["exact_one"][view_index] = exact_depth
                predicted[event_id]["exact_two"][view_index] = exact_two_depth
                if not (
                    output_shape == generic_shape == exact_shape == exact_two_shape
                ):
                    raise RuntimeError("EXP-050 output shapes differ across policies")
                parity_max = max(
                    parity_max,
                    float((base_points - exact["base_points"]).abs().max()),
                )
                adapted_query_frames += 1
            previous_points = base_points.detach()
            state = next_state
            if processed_frames % 100 == 0 or position == len(views) - 1:
                print(
                    json.dumps(
                        {
                            "sequence": sequence_number + 1,
                            "sequences": len(run_sequences),
                            "processed_frames": processed_frames,
                            "adapted_query_frames": adapted_query_frames,
                        }
                    ),
                    flush=True,
                )
            del base_prediction, auxiliary
            if processed_frames % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        del images, views, state, previous_points
        gc.collect()
        torch.cuda.empty_cache()

    _copy_weight(carrier, exact_weight)
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    baseline_by_id = {row["target"]: row for row in baseline["rows"]}
    rows = []
    for event_id in sorted(permitted):
        event = event_by_id[event_id]
        buffers = predicted[event_id]
        if not all(all(value is not None for value in buffers[policy]) for policy in POLICIES):
            raise RuntimeError(f"EXP-050 incomplete query prediction: {event_id}")
        target, valid = _query_depth_gt(event, side, config)
        height, width = output_shapes[event_id]
        original_width, original_height = Image.open(event["query"][0]["rgb"]).size
        fx, fy, cx, cy = event["intrinsics_fx_fy_cx_cy"]
        intrinsics = np.tile(
            np.asarray(
                [
                    fx * width / original_width,
                    fy * height / original_height,
                    cx * width / original_width,
                    cy * height / original_height,
                ],
                dtype=np.float64,
            ),
            (4, 1),
        )
        policy_metrics = {}
        for policy in POLICIES:
            metrics = _depth_metrics(
                np.stack(buffers[policy]),
                target,
                valid,
                intrinsics,
                image_size=(height, width),
                minimum_cells=int(config["depth"]["minimum_cells_per_view"]),
            )
            if metrics is None:
                raise RuntimeError(f"EXP-050 has no valid metrics: {policy}:{event_id}")
            policy_metrics[policy] = metrics
        reference = baseline_by_id[event_id]
        rows.append(
            {
                "target": event_id,
                "sequence": event["sequence"],
                **policy_metrics,
                "ttt3r": reference["ttt3r"],
                "exp036_cut3r": reference["cut3r"],
            }
        )

    if args.smoke_targets:
        print(
            json.dumps(
                {
                    "smoke_targets": len(rows),
                    "processed_frames": processed_frames,
                    "adapted_query_frames": adapted_query_frames,
                    "passed": True,
                }
            )
        )
        return

    summaries = {
        policy: _summary(rows, policy)
        for policy in (*POLICIES, "ttt3r", "exp036_cut3r")
    }
    comparisons = {
        "exact_one_over_cut3r": ("cut3r", "exact_one"),
        "exact_one_over_generic": ("generic_one", "exact_one"),
        "exact_one_over_ttt3r": ("ttt3r", "exact_one"),
        "exact_two_over_exact_one": ("exact_one", "exact_two"),
        "generic_one_over_cut3r": ("cut3r", "generic_one"),
    }
    uncertainty = _bootstrap(rows, comparisons, config)
    reproduction_max = max(
        abs(row["cut3r"][metric] - row["exp036_cut3r"][metric])
        for row in rows
        for metric in METRICS
    )
    method_checks = {
        f"exact_meta_better_cut3r_{metric}_ci95": uncertainty[
            "exact_one_over_cut3r"
        ][metric]["ci95"][0]
        > 0
        for metric in PRIMARY
    }
    learned_checks = {
        f"exact_meta_better_generic_{metric}_ci95": uncertainty[
            "exact_one_over_generic"
        ][metric]["ci95"][0]
        > 0
        for metric in PRIMARY
    }
    competitive_checks = {
        f"exact_meta_better_ttt3r_{metric}_ci95": uncertainty[
            "exact_one_over_ttt3r"
        ][metric]["ci95"][0]
        > 0
        for metric in PRIMARY
    }
    common_checks = {
        "exact_coverage": len(rows) == int(config["data"]["exact_targets"])
        and len({row["sequence"] for row in rows})
        == int(config["data"]["exact_sequences"])
        and processed_frames == int(config["data"]["exact_stream_frames"])
        and adapted_query_frames == 4 * int(config["data"]["exact_targets"]),
        "finite": all(
            math.isfinite(summary[metric])
            for summary in summaries.values()
            for metric in METRICS
        ),
        "exact_cached_readout_parity": parity_max == 0.0,
        "base_reproduction": reproduction_max
        <= float(config["success"]["maximum_base_reproduction_abs_difference"]),
    }
    method_passed = all({**common_checks, **method_checks, **learned_checks}.values())
    competitive_passed = method_passed and all(competitive_checks.values())
    result = {
        "experiment": "EXP-050",
        "stage": config["purpose"],
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "targets": len(rows),
        "sequences": len({row["sequence"] for row in rows}),
        "processed_frames": processed_frames,
        "adapted_query_frames": adapted_query_frames,
        "summaries": summaries,
        "uncertainty": uncertainty,
        "base_reproduction_max_abs": reproduction_max,
        "cached_readout_parity_max_abs": parity_max,
        "runtime": {
            "seconds_including_preprocessing": runtime_seconds,
            "peak_allocated_gib": peak_memory_gib,
        },
        "registered_gate": {
            "common_checks": common_checks,
            "method_checks": {**method_checks, **learned_checks},
            "competitive_checks": competitive_checks,
            "method_feasibility_passed": method_passed,
            "top_tier_competitiveness_passed": competitive_passed,
        },
        "fitting_performed": False,
        "memory_active": False,
        "query_recurrent_state_update": False,
        "query_local_code_persistent": False,
        "rgbd_used_for_adaptation": False,
        "tum_exposed_development": True,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "summaries": summaries,
                "uncertainty": uncertainty,
                "reproduction_max": reproduction_max,
                "runtime": result["runtime"],
                "gate": result["registered_gate"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
