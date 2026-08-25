#!/usr/bin/env python3
"""Fit and audit the single EXP-042 CUT3R plasticity basis."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
    transport_code_visual,
)
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.evaluate_exp040_cut3r_oracle_reuse_premise import _scene_balanced
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _ordered_scenes(rows: list[dict]) -> list[str]:
    scenes: list[str] = []
    for row in rows:
        if row["scene"] not in scenes:
            scenes.append(row["scene"])
    return scenes


def _subset(
    rows: list[dict], *, scene_offset: int, scene_count: int, pairs_per_scene: int
) -> list[dict]:
    scenes = _ordered_scenes(rows)[scene_offset : scene_offset + scene_count]
    selected: list[dict] = []
    for scene in scenes:
        selected.extend([row for row in rows if row["scene"] == scene][:pairs_per_scene])
    return selected


def _pair_ids_sha256(rows: list[dict]) -> str:
    payload = json.dumps(
        [row["pair_id"] for row in rows], separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _points(prediction: dict, patch_size: int) -> torch.Tensor:
    return patch_center_points(prediction["pts3d_in_other_view"], patch_size)


def _online_code(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    previous_points: torch.Tensor,
    *,
    patch_size: int,
    normalized_step: float,
) -> dict:
    tokens = auxiliary["decoder_patch_tokens"].shape[1]
    code = torch.zeros(
        1, tokens, carrier.code_dim, device=previous_points.device, requires_grad=True
    )
    prediction = carrier.readout(auxiliary, code=code)
    points = _points(prediction, patch_size)
    loss = symmetric_point_consistency(points, previous_points)
    gradient = torch.autograd.grad(loss, code, create_graph=False)[0]
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    updated = code.detach() - float(normalized_step) * normalized.detach()
    return {
        "base_points": points.detach(),
        "base_loss": float(loss.detach()),
        "code": updated,
        "gradient_norm": float(gradient.norm()),
    }


def _prepare_pair(
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
    source = _online_code(
        carrier,
        source_auxiliary,
        source_previous_points,
        patch_size=patch_size,
        normalized_step=normalized_step,
    )
    with torch.no_grad():
        target_previous, target_state, _ = carrier.step(views[2], source_state)
        target_previous_points = _points(target_previous, patch_size)
        target_base, _, target_auxiliary = carrier.step(views[3], target_state)
    target = _online_code(
        carrier,
        target_auxiliary,
        target_previous_points,
        patch_size=patch_size,
        normalized_step=normalized_step,
    )
    # The code-free readout from step and the zero-code readout must stay the
    # same official prediction. Retain a numerical guard during fitting/audit.
    parity = float(
        (_points(source_base, patch_size) - source["base_points"]).abs().max()
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
        "transported": transported.detach(),
        "transport_peak": float(peak.mean()),
        "target_previous_points": target_previous_points.detach(),
        "target_auxiliary": target_auxiliary,
        "parity": parity,
    }


def _loss_for_code(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    code: torch.Tensor,
    previous_points: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    return symmetric_point_consistency(
        _points(carrier.readout(auxiliary, code=code), patch_size), previous_points
    )


def _alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean()
    )


def _evaluate_basis(
    carrier: FrozenCUT3RCarrier,
    rows: list[dict],
    load_images_for_eval,
    config: dict,
    *,
    label: str,
) -> dict:
    patch_size = int(config["plasticity"]["patch_size"])
    normalized_step = float(config["plasticity"]["normalized_step"])
    carrier.residual.requires_grad_(False)
    results = []
    for index, row in enumerate(rows):
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
        prepared = _prepare_pair(
            carrier,
            views,
            patch_size=patch_size,
            normalized_step=normalized_step,
            visual_temperature=float(config["transport"]["visual_temperature"]),
        )
        target = prepared["target"]
        transported = prepared["transported"]
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["transport"]["spatial_shuffle_seed"]) + index
        )
        permutation = torch.randperm(transported.shape[1], generator=generator).to(
            transported.device
        )
        with torch.no_grad():
            current_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"],
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
            full_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + transported,
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
            shuffle_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + transported[:, permutation],
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
        results.append(
            {
                "pair_id": row["pair_id"],
                "scene": row["scene"],
                "target_base_loss": target["base_loss"],
                "target_current_loss": current_loss,
                "target_full_loss": full_loss,
                "target_visual_shuffle_loss": shuffle_loss,
                "source_target_code_agreement": _alignment(transported, target["code"]),
                "visual_mean_peak_weight": prepared["transport_peak"],
                "cached_readout_parity_max_abs": prepared["parity"],
            }
        )
        print(
            json.dumps(
                {"audit_basis": label, "evaluated": index + 1, "total": len(rows)}
            ),
            flush=True,
        )
        del images, views, prepared
        gc.collect()
        torch.cuda.empty_cache()

    keys = (
        "target_base_loss",
        "target_current_loss",
        "target_full_loss",
        "target_visual_shuffle_loss",
        "source_target_code_agreement",
        "visual_mean_peak_weight",
        "cached_readout_parity_max_abs",
    )
    means = {key: _scene_balanced(results, key) for key in keys}
    gains = {
        "current_ttt": means["target_base_loss"] - means["target_current_loss"],
        "oracle_reuse_over_current": means["target_current_loss"]
        - means["target_full_loss"],
        "full_over_visual_spatial_shuffle": means["target_visual_shuffle_loss"]
        - means["target_full_loss"],
    }
    harm = float(
        np.mean(
            [row["target_full_loss"] > row["target_current_loss"] for row in results]
        )
    )
    return {"means": means, "gains": gains, "reuse_harm_fraction": harm, "rows": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-042_learned_cut3r_plasticity_coordinate_v10.yaml"
    )
    parser.add_argument("--confirm-train-only-fit", action="store_true")
    parser.add_argument("--smoke-pairs", type=int, default=0)
    args = parser.parse_args()
    if not args.confirm_train_only_fit:
        raise SystemExit("EXP-042 requires explicit train-only confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-042 requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    checkpoint_output = Path(config["output"]["checkpoint"])
    official = args.smoke_pairs == 0
    if official and (output.exists() or checkpoint_output.exists()):
        raise RuntimeError("EXP-042 official artifact already exists")

    manifest_path = Path(config["data"]["manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-042 source-safe contract failed")

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
        raise RuntimeError("EXP-042 internal split contract failed")
    if args.smoke_pairs:
        fit_rows = fit_rows[: args.smoke_pairs]
        audit_rows = audit_rows[: args.smoke_pairs]

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
    normalized_step = float(config["plasticity"]["normalized_step"])
    trace = []
    for index, row in enumerate(fit_rows):
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
        prepared = _prepare_pair(
            carrier,
            views,
            patch_size=patch_size,
            normalized_step=normalized_step,
            visual_temperature=float(config["transport"]["visual_temperature"]),
        )
        optimizer.zero_grad(set_to_none=True)
        current_loss = _loss_for_code(
            carrier,
            prepared["target_auxiliary"],
            prepared["target"]["code"],
            prepared["target_previous_points"],
            patch_size,
        )
        (float(config["training"]["current_loss_weight"]) * current_loss).backward()
        full_loss = _loss_for_code(
            carrier,
            prepared["target_auxiliary"],
            prepared["target"]["code"] + prepared["transported"],
            prepared["target_previous_points"],
            patch_size,
        )
        (float(config["training"]["oracle_reuse_loss_weight"]) * full_loss).backward()
        gradient_norm = float(
            torch.linalg.vector_norm(
                torch.cat(
                    [parameter.grad.flatten() for parameter in carrier.residual.parameters()]
                )
            )
        )
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
                    "total": len(fit_rows),
                    "current": trace[-1]["current_loss"],
                    "reuse": trace[-1]["oracle_reuse_loss"],
                }
            ),
            flush=True,
        )
        del images, views, prepared, current_loss, full_loss
        gc.collect()
        torch.cuda.empty_cache()

    learned_basis = deepcopy(carrier.residual.state_dict())
    learned_weight = carrier.residual.projection.weight.detach().clone()
    basis_change = float((learned_weight - initial_weight).norm())

    # The audit is first opened only after the fixed fit has completed. Both
    # deterministic initial and learned bases are then measured without
    # selecting or modifying either.
    carrier.residual.load_state_dict(initial_basis)
    initial_audit = _evaluate_basis(
        carrier, audit_rows, load_images_for_eval, config, label="initial"
    )
    carrier.residual.load_state_dict(learned_basis)
    learned_audit = _evaluate_basis(
        carrier, audit_rows, load_images_for_eval, config, label="learned"
    )
    checkpoint_payload = {
        "experiment": "EXP-042",
        "protocol_revision": config["protocol_revision"],
        "residual_state_dict": learned_basis,
        "fit_pair_ids_sha256": config["data"]["fit_pair_ids_sha256"],
        "audit_pair_ids_sha256": config["data"]["audit_pair_ids_sha256"],
        "basis_change_l2": basis_change,
    }
    if official:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, checkpoint_output)
        checkpoint_sha256 = _sha256(checkpoint_output)
    else:
        checkpoint_sha256 = None

    learned = learned_audit
    checks = {
        "exact_coverage": len(fit_rows) == int(config["success"]["exact_fit_pairs"])
        and len({row["scene"] for row in fit_rows})
        == int(config["success"]["exact_fit_scenes"])
        and len(learned["rows"]) == int(config["success"]["exact_audit_pairs"])
        and len({row["scene"] for row in learned["rows"]})
        == int(config["success"]["exact_audit_scenes"]),
        "finite": all(
            math.isfinite(value)
            for audit in (initial_audit, learned_audit)
            for value in (
                list(audit["means"].values())
                + list(audit["gains"].values())
                + [audit["reuse_harm_fraction"]]
            )
        )
        and all(
            math.isfinite(row["basis_gradient_norm"])
            and math.isfinite(row["current_loss"])
            and math.isfinite(row["oracle_reuse_loss"])
            for row in trace
        ),
        "basis_changed": basis_change > 0,
        "positive_current_ttt_gain": learned["gains"]["current_ttt"] > 0,
        "positive_oracle_reuse_gain_over_current": learned["gains"][
            "oracle_reuse_over_current"
        ]
        > 0,
        "full_better_visual_spatial_shuffle": learned["gains"][
            "full_over_visual_spatial_shuffle"
        ]
        > 0,
        "positive_source_target_code_agreement": learned["means"][
            "source_target_code_agreement"
        ]
        > 0,
        "reuse_harm_below_limit": learned["reuse_harm_fraction"]
        <= float(config["success"]["maximum_reuse_harm_fraction"]),
        "reuse_gain_better_than_initial_basis": learned["gains"][
            "oracle_reuse_over_current"
        ]
        > initial_audit["gains"]["oracle_reuse_over_current"],
    }
    result = {
        "experiment": "EXP-042",
        "stage": "learned_cut3r_plasticity_coordinate_oracle_premise",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "official": official,
        "carrier_frozen": True,
        "fitted_parameters": sum(
            parameter.numel() for parameter in carrier.residual.parameters()
        ),
        "optimizer_steps": len(trace),
        "basis_change_l2": basis_change,
        "checkpoint": str(checkpoint_output) if official else None,
        "checkpoint_sha256": checkpoint_sha256,
        "fit_scenes": len({row["scene"] for row in fit_rows}),
        "fit_pairs": len(fit_rows),
        "audit_scenes": len({row["scene"] for row in audit_rows}),
        "audit_pairs": len(audit_rows),
        "initial_basis_audit": initial_audit,
        "learned_basis_audit": learned_audit,
        "fit_trace": trace,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "address_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
    }
    if official:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "basis_change_l2": basis_change,
                "initial": {
                    "means": initial_audit["means"],
                    "gains": initial_audit["gains"],
                    "harm": initial_audit["reuse_harm_fraction"],
                },
                "learned": {
                    "means": learned_audit["means"],
                    "gains": learned_audit["gains"],
                    "harm": learned_audit["reuse_harm_fraction"],
                },
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
