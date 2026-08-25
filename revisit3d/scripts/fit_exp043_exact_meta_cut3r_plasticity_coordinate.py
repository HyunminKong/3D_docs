#!/usr/bin/env python3
"""Fit EXP-043 by differentiating through both online code updates."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    symmetric_point_consistency,
    transport_code_visual,
)
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp042_learned_cut3r_plasticity_coordinate import (
    _evaluate_basis,
    _loss_for_code,
    _pair_ids_sha256,
    _points,
    _subset,
)


def _exact_online_code(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    previous_points: torch.Tensor,
    *,
    patch_size: int,
    normalized_step: float,
) -> dict:
    """Create one code while retaining its dependence on the shared basis."""
    tokens = auxiliary["decoder_patch_tokens"].shape[1]
    code = torch.zeros(
        1, tokens, carrier.code_dim, device=previous_points.device, requires_grad=True
    )
    prediction = carrier.readout(auxiliary, code=code)
    points = _points(prediction, patch_size)
    loss = symmetric_point_consistency(points, previous_points)
    gradient = torch.autograd.grad(loss, code, create_graph=True)[0]
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    updated = code - float(normalized_step) * normalized
    return {
        "base_points": points.detach(),
        "base_loss": float(loss.detach()),
        "code": updated,
        "gradient_norm": float(gradient.detach().norm()),
    }


def _prepare_exact_pair(
    carrier: FrozenCUT3RCarrier,
    views: list[dict],
    *,
    patch_size: int,
    normalized_step: float,
    visual_temperature: float,
) -> dict:
    with torch.no_grad():
        source_previous, state, _ = carrier.step(views[0], None)
        source_previous_points = _points(source_previous, patch_size)
        source_base, source_state, source_auxiliary = carrier.step(views[1], state)
    source = _exact_online_code(
        carrier,
        source_auxiliary,
        source_previous_points,
        patch_size=patch_size,
        normalized_step=normalized_step,
    )
    with torch.no_grad():
        target_previous, target_state, _ = carrier.step(views[2], source_state)
        target_previous_points = _points(target_previous, patch_size)
        _, _, target_auxiliary = carrier.step(views[3], target_state)
    target = _exact_online_code(
        carrier,
        target_auxiliary,
        target_previous_points,
        patch_size=patch_size,
        normalized_step=normalized_step,
    )
    transported, peak = transport_code_visual(
        source_auxiliary["image_tokens"],
        source["code"],
        target_auxiliary["image_tokens"],
        temperature=visual_temperature,
    )
    return {
        "source": source,
        "target": target,
        "transported": transported,
        "transport_peak": float(peak.mean()),
        "target_previous_points": target_previous_points.detach(),
        "target_auxiliary": target_auxiliary,
        "parity": float(
            (_points(source_base, patch_size) - source["base_points"]).abs().max()
        ),
    }


def _scene_values(rows: list[dict], left: str, right: str | None = None) -> np.ndarray:
    scenes = sorted({row["scene"] for row in rows})
    values = []
    for scene in scenes:
        selected = [row for row in rows if row["scene"] == scene]
        if right is None:
            values.append(np.mean([row[left] for row in selected]))
        else:
            values.append(np.mean([row[left] - row[right] for row in selected]))
    return np.asarray(values, dtype=np.float64)


def _bootstrap_summary(
    rows: list[dict], *, draws: int, seed: int
) -> dict[str, dict[str, float | list[float] | int]]:
    metrics = {
        "current_ttt_gain": _scene_values(
            rows, "target_base_loss", "target_current_loss"
        ),
        "oracle_reuse_gain_over_current": _scene_values(
            rows, "target_current_loss", "target_full_loss"
        ),
        "full_over_visual_spatial_shuffle": _scene_values(
            rows, "target_visual_shuffle_loss", "target_full_loss"
        ),
    }
    summary = {}
    for offset, (name, values) in enumerate(metrics.items()):
        generator = np.random.default_rng(seed + offset)
        indices = generator.integers(0, len(values), size=(draws, len(values)))
        bootstrap = values[indices].mean(axis=1)
        summary[name] = {
            "mean": float(values.mean()),
            "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "positive_scenes": int((values > 0).sum()),
            "scenes": len(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-043_exact_meta_cut3r_plasticity_coordinate_v10.yaml"
    )
    parser.add_argument("--confirm-train-only-fit", action="store_true")
    parser.add_argument("--fit-only-smoke", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_only_fit:
        raise SystemExit("EXP-043 requires explicit train-only confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-043 requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    checkpoint_output = Path(config["output"]["checkpoint"])
    if not args.fit_only_smoke and (output.exists() or checkpoint_output.exists()):
        raise RuntimeError("EXP-043 official artifact already exists")
    manifest_path = Path(config["data"]["manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-043 source-safe contract failed")

    rows = json.loads(manifest_path.read_text())
    fit_rows = _subset(
        rows,
        scene_offset=int(config["data"]["fit_scene_offset"]),
        scene_count=int(config["data"]["fit_scenes"]),
        pairs_per_scene=int(config["data"]["pairs_per_scene"]),
    )
    audit_rows = _subset(
        rows,
        scene_offset=int(config["data"]["audit_scene_offset"]),
        scene_count=int(config["data"]["audit_scenes"]),
        pairs_per_scene=int(config["data"]["pairs_per_scene"]),
    )
    if not (
        len(fit_rows) == int(config["data"]["fit_pairs"])
        and len(audit_rows) == int(config["data"]["audit_pairs"])
        and _pair_ids_sha256(fit_rows) == config["data"]["fit_pair_ids_sha256"]
        and _pair_ids_sha256(audit_rows) == config["data"]["audit_pair_ids_sha256"]
        and {row["scene"] for row in fit_rows}.isdisjoint(
            {row["scene"] for row in audit_rows}
        )
    ):
        raise RuntimeError("EXP-043 internal split contract failed")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # CUT3R's default memory-efficient SDPA kernel has no double backward in
    # this PyTorch build. The math backend computes the identical attention
    # operation with the derivative required by the registered exact meta-step.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    carrier = FrozenCUT3RCarrier(
        carrier_checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
    ).cuda()
    carrier.eval()
    initial_basis = deepcopy(carrier.residual.state_dict())
    initial_weight = carrier.residual.projection.weight.detach().clone()
    carrier.residual.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        carrier.residual.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    patch_size = int(config["plasticity"]["patch_size"])
    fit_sequence = fit_rows[:1] if args.fit_only_smoke else fit_rows
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for index, row in enumerate(fit_sequence):
        optimizer.zero_grad(set_to_none=True)
        images = load_images_for_eval(
            [
                row[key]
                for key in (
                    "source_previous_rgb",
                    "source_rgb",
                    "target_previous_rgb",
                    "target_rgb",
                )
            ],
            size=int(config["carrier"]["image_size"]),
            verbose=False,
            crop=bool(config["carrier"]["crop"]),
        )
        views = _views(images, [True, True, True, True])
        prepared = _prepare_exact_pair(
            carrier,
            views,
            patch_size=patch_size,
            normalized_step=float(config["plasticity"]["normalized_step"]),
            visual_temperature=float(config["transport"]["visual_temperature"]),
        )
        current_loss = _loss_for_code(
            carrier,
            prepared["target_auxiliary"],
            prepared["target"]["code"],
            prepared["target_previous_points"],
            patch_size,
        )
        full_loss = _loss_for_code(
            carrier,
            prepared["target_auxiliary"],
            prepared["target"]["code"] + prepared["transported"],
            prepared["target_previous_points"],
            patch_size,
        )
        objective = (
            float(config["training"]["current_loss_weight"]) * current_loss
            + float(config["training"]["oracle_reuse_loss_weight"]) * full_loss
        )
        objective.backward()
        gradient = torch.cat(
            [parameter.grad.flatten() for parameter in carrier.residual.parameters()]
        )
        gradient_norm = float(gradient.norm())
        optimizer.step()
        trace.append(
            {
                "step": index + 1,
                "scene": row["scene"],
                "pair_id": row["pair_id"],
                "current_loss": float(current_loss.detach()),
                "oracle_reuse_loss": float(full_loss.detach()),
                "basis_gradient_norm": gradient_norm,
                "cached_readout_parity_max_abs": prepared["parity"],
            }
        )
        print(
            json.dumps(
                {
                    "fit": index + 1,
                    "total": len(fit_sequence),
                    "current": trace[-1]["current_loss"],
                    "reuse": trace[-1]["oracle_reuse_loss"],
                    "gradient_norm": gradient_norm,
                    "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            ),
            flush=True,
        )
        del images, views, prepared, current_loss, full_loss, objective, gradient
        gc.collect()
        torch.cuda.empty_cache()

    learned_basis = deepcopy(carrier.residual.state_dict())
    learned_weight = carrier.residual.projection.weight.detach().clone()
    basis_change = float((learned_weight - initial_weight).norm())
    peak_cuda_gib = float(torch.cuda.max_memory_allocated() / 2**30)
    if args.fit_only_smoke:
        smoke = {
            "experiment": "EXP-043",
            "fit_only_smoke": True,
            "fit_pairs_accessed": 1,
            "audit_pairs_accessed": 0,
            "basis_change_l2": basis_change,
            "peak_cuda_gib": peak_cuda_gib,
            "finite": all(
                math.isfinite(value)
                for value in (
                    trace[0]["current_loss"],
                    trace[0]["oracle_reuse_loss"],
                    trace[0]["basis_gradient_norm"],
                    basis_change,
                )
            ),
        }
        print(json.dumps(smoke, indent=2))
        return

    # The new audit is opened only after the fixed exact-meta fit is complete.
    carrier.residual.load_state_dict(initial_basis)
    initial_audit = _evaluate_basis(
        carrier, audit_rows, load_images_for_eval, config, label="initial"
    )
    carrier.residual.load_state_dict(learned_basis)
    learned_audit = _evaluate_basis(
        carrier, audit_rows, load_images_for_eval, config, label="learned"
    )
    draws = int(config["audit"]["bootstrap_draws"])
    bootstrap_seed = int(config["audit"]["bootstrap_seed"])
    initial_uncertainty = _bootstrap_summary(
        initial_audit["rows"], draws=draws, seed=bootstrap_seed
    )
    learned_uncertainty = _bootstrap_summary(
        learned_audit["rows"], draws=draws, seed=bootstrap_seed
    )

    checkpoint_payload = {
        "experiment": "EXP-043",
        "protocol_revision": config["protocol_revision"],
        "residual_state_dict": learned_basis,
        "fit_pair_ids_sha256": config["data"]["fit_pair_ids_sha256"],
        "audit_pair_ids_sha256": config["data"]["audit_pair_ids_sha256"],
        "basis_change_l2": basis_change,
    }
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, checkpoint_output)
    checkpoint_sha256 = _sha256(checkpoint_output)
    learned = learned_audit
    checks = {
        "exact_coverage": len(fit_sequence) == int(config["success"]["exact_fit_pairs"])
        and len({row["scene"] for row in fit_sequence})
        == int(config["success"]["exact_fit_scenes"])
        and len(learned["rows"]) == int(config["success"]["exact_audit_pairs"])
        and len({row["scene"] for row in learned["rows"]})
        == int(config["success"]["exact_audit_scenes"]),
        "finite": all(
            math.isfinite(value)
            for audit in (initial_audit, learned_audit)
            for value in list(audit["means"].values())
            + list(audit["gains"].values())
            + [audit["reuse_harm_fraction"]]
        )
        and all(
            math.isfinite(row["basis_gradient_norm"])
            and math.isfinite(row["current_loss"])
            and math.isfinite(row["oracle_reuse_loss"])
            for row in trace
        ),
        "basis_changed": basis_change > 0,
        "exact_cached_readout_parity": max(
            [row["cached_readout_parity_max_abs"] for row in trace]
            + [
                row["cached_readout_parity_max_abs"]
                for audit in (initial_audit, learned_audit)
                for row in audit["rows"]
            ]
        )
        == 0,
        "positive_current_ttt_gain_ci95": learned_uncertainty["current_ttt_gain"][
            "ci95"
        ][0]
        > 0,
        "positive_oracle_reuse_gain_ci95": learned_uncertainty[
            "oracle_reuse_gain_over_current"
        ]["ci95"][0]
        > 0,
        "positive_full_over_visual_shuffle_ci95": learned_uncertainty[
            "full_over_visual_spatial_shuffle"
        ]["ci95"][0]
        > 0,
        "reuse_harm_below_limit": learned["reuse_harm_fraction"]
        <= float(config["success"]["maximum_reuse_harm_fraction"]),
        "reuse_gain_better_than_initial_basis": learned["gains"][
            "oracle_reuse_over_current"
        ]
        > initial_audit["gains"]["oracle_reuse_over_current"],
    }
    result = {
        "experiment": "EXP-043",
        "stage": "exact_meta_cut3r_plasticity_coordinate_oracle_premise",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "carrier_frozen": True,
        "fitted_parameters": sum(
            parameter.numel() for parameter in carrier.residual.parameters()
        ),
        "optimizer_steps": len(trace),
        "basis_change_l2": basis_change,
        "peak_cuda_gib": peak_cuda_gib,
        "checkpoint": str(checkpoint_output),
        "checkpoint_sha256": checkpoint_sha256,
        "fit_scenes": len({row["scene"] for row in fit_sequence}),
        "fit_pairs": len(fit_sequence),
        "audit_scenes": len({row["scene"] for row in audit_rows}),
        "audit_pairs": len(audit_rows),
        "initial_basis_audit": initial_audit,
        "learned_basis_audit": learned_audit,
        "initial_basis_uncertainty": initial_uncertainty,
        "learned_basis_uncertainty": learned_uncertainty,
        "fit_trace": trace,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "address_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "basis_change_l2": basis_change,
                "peak_cuda_gib": peak_cuda_gib,
                "initial": {
                    "gains": initial_audit["gains"],
                    "harm": initial_audit["reuse_harm_fraction"],
                    "uncertainty": initial_uncertainty,
                },
                "learned": {
                    "gains": learned_audit["gains"],
                    "harm": learned_audit["reuse_harm_fraction"],
                    "uncertainty": learned_uncertainty,
                },
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
