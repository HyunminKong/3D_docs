#!/usr/bin/env python3
"""Fit and train-scene-audit the single EXP-055 token-axis conditioner."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    TokenAxisConditioner,
    patch_center_points,
    symmetric_point_consistency,
)
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _rms_normalize,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp053_exact_metric_aligned_ttt3r_basis import (
    _cache,
    _ci,
    _metric,
    _tuple_specs,
)


def _audit(carrier, conditioner, dataset, specs, config) -> list[dict]:
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    rows = []
    for index, spec in enumerate(specs):
        gt_views = dataset[index]
        base, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
        tokens = auxiliary["decoder_patch_tokens"].float()
        scales = conditioner(tokens)
        zero = torch.zeros(
            1, tokens.shape[1], carrier.code_dim, device="cuda", requires_grad=True
        )
        zero_prediction = carrier.readout(auxiliary, code=zero, axis_scale=scales)
        parity = max(
            float((zero_prediction[key].detach() - base[key].detach()).abs().max())
            for key in ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")
            if key in zero_prediction and key in base
        )
        before_online = symmetric_point_consistency(
            patch_center_points(zero_prediction["pts3d_in_other_view"], patch_size),
            previous_points,
        )
        before_metric = _metric(zero_prediction, gt_views[-1], config)
        gradient = torch.autograd.grad(before_online, zero, create_graph=False)[0]
        code = zero.detach() - step_size * _rms_normalize(gradient.detach())
        with torch.no_grad():
            adapted = carrier.readout(
                auxiliary, code=code, axis_scale=scales.detach()
            )
            after_online = symmetric_point_consistency(
                patch_center_points(adapted["pts3d_in_other_view"], patch_size),
                previous_points,
            )
            after_metric = _metric(adapted, gt_views[-1], config)
        sequence = spec.split()[0]
        rows.append(
            {
                "sequence": sequence,
                "scene": sequence.split("/", 1)[0],
                "target_frame": int(spec.split()[-1]),
                "zero_code_max_abs_difference": parity,
                "online_loss_gain": float(before_online.detach() - after_online),
                "metric_gain": float(before_metric.detach() - after_metric),
                "metric_harm": bool(after_metric > before_metric),
                "scale_mean": float(scales.detach().mean()),
                "scale_std": float(scales.detach().std()),
                "scale_min": float(scales.detach().min()),
                "scale_max": float(scales.detach().max()),
            }
        )
        del gt_views, base, auxiliary, previous_points, tokens, scales, zero
        del zero_prediction, before_online, before_metric, gradient, code
        del adapted, after_online, after_metric
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _row_map(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["sequence"], int(row["target_frame"])): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-055_learned_conditional_tangent_v10.yaml"
    )
    parser.add_argument("--confirm-train-only-fit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_only_fit or not torch.cuda.is_available():
        raise SystemExit("EXP-055 requires train-only fit confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    checkpoint_output = Path(config["output"]["checkpoint"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    global_result_path = Path(config["controls"]["learned_global_result"])
    if output.exists() or checkpoint_output.exists():
        raise RuntimeError("EXP-055 artifact already exists")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    global_result = json.loads(global_result_path.read_text())
    fit_scenes = {value.split("/", 1)[0] for value in config["data"]["fit_sequences"]}
    audit_scenes = {
        value.split("/", 1)[0] for value in config["data"]["audit_sequences"]
    }
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(global_result_path)
        == config["controls"]["learned_global_result_sha256"]
        and global_result["experiment"] == "EXP-053"
        and fit_scenes.isdisjoint(audit_scenes)
        and len(fit_scenes) == int(config["success"]["exact_fit_scenes"])
        and len(audit_scenes) == int(config["success"]["exact_audit_scenes"])
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and config["plasticity"]["frozen_basis"] is True
        and config["training"]["fitted_parameters"]
        == "token_axis_conditioner_only"
    ):
        raise RuntimeError("EXP-055 frozen contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "src"))
    from eval.mv_recon.data import SevenScenes

    fit_specs = _tuple_specs(
        config["data"]["fit_sequences"],
        [int(value) for value in config["data"]["target_frames"]],
        int(config["data"]["context_frames"]),
    )
    audit_specs = _tuple_specs(
        config["data"]["audit_sequences"],
        [int(value) for value in config["data"]["target_frames"]],
        int(config["data"]["context_frames"]),
    )
    if not (
        len(fit_specs) == int(config["success"]["exact_fit_anchors"])
        and len(audit_specs) == int(config["success"]["exact_audit_anchors"])
    ):
        raise RuntimeError("EXP-055 anchor coverage differs from registration")
    dataset_args = {
        "split": "train",
        "ROOT": config["data"]["root"],
        "resolution": tuple(config["carrier"]["resolution"]),
        "seed": seed,
    }
    fit_dataset = SevenScenes(tuple_list=fit_specs, **dataset_args)
    audit_dataset = SevenScenes(tuple_list=audit_specs, **dataset_args)
    carrier = FrozenCUT3RCarrier(
        carrier_checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
        update_type="ttt3r",
    ).cuda()
    carrier.eval()
    carrier.model.requires_grad_(False)
    carrier.residual.requires_grad_(False)
    conditioner = TokenAxisConditioner(
        token_dim=int(carrier.model.dec_embed_dim), code_dim=carrier.code_dim
    ).cuda()
    conditioner.train()
    initial_state = deepcopy(conditioner.state_dict())
    initial_weight = conditioner.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        conditioner.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    torch.cuda.reset_peak_memory_stats()
    initial_rows = _audit(carrier, conditioner, audit_dataset, audit_specs, config)
    trace = []
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    for index, spec in enumerate(fit_specs):
        optimizer.zero_grad(set_to_none=True)
        gt_views = fit_dataset[index]
        _, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
        tokens = auxiliary["decoder_patch_tokens"].float()
        scales = conditioner(tokens)
        zero = torch.zeros(
            1, tokens.shape[1], carrier.code_dim, device="cuda", requires_grad=True
        )
        prediction = carrier.readout(auxiliary, code=zero, axis_scale=scales)
        online_loss = symmetric_point_consistency(
            patch_center_points(prediction["pts3d_in_other_view"], patch_size),
            previous_points,
        )
        online_gradient = torch.autograd.grad(
            online_loss, zero, create_graph=True, retain_graph=True
        )[0]
        code = zero - step_size * _rms_normalize(online_gradient)
        adapted = carrier.readout(auxiliary, code=code, axis_scale=scales)
        outer = _metric(adapted, gt_views[-1], config)
        outer.backward()
        gradient_norm = float(conditioner.projection.weight.grad.norm())
        optimizer.step()
        trace.append(
            {
                "step": index + 1,
                "sequence": spec.split()[0],
                "target_frame": int(spec.split()[-1]),
                "outer_metric": float(outer.detach()),
                "conditioner_gradient_norm": gradient_norm,
                "scale_mean": float(scales.detach().mean()),
                "scale_std": float(scales.detach().std()),
            }
        )
        print(json.dumps({"fit": index + 1, "total": len(fit_specs), **trace[-1]}), flush=True)
        del gt_views, auxiliary, previous_points, tokens, scales, zero, prediction
        del online_loss, online_gradient, code, adapted, outer
        gc.collect()
        torch.cuda.empty_cache()

    final_rows = _audit(carrier, conditioner, audit_dataset, audit_specs, config)
    initial_map = _row_map(initial_rows)
    exp053_initial_map = _row_map(global_result["initial_rows"])
    exp053_final_map = _row_map(global_result["final_rows"])
    if set(initial_map) != set(exp053_initial_map) or set(initial_map) != set(exp053_final_map):
        raise RuntimeError("EXP-055/EXP-053 audit row identities differ")
    initial_reproduction_max_abs = max(
        abs(initial_map[key][metric] - exp053_initial_map[key][metric])
        for key in initial_map
        for metric in ("online_loss_gain", "metric_gain")
    )

    initial_gain = np.asarray([row["metric_gain"] for row in initial_rows])
    final_gain = np.asarray([row["metric_gain"] for row in final_rows])
    paired_gain = final_gain - initial_gain
    global_final_gain = np.asarray(
        [exp053_final_map[(row["sequence"], row["target_frame"])]["metric_gain"] for row in final_rows]
    )
    conditional_minus_global = final_gain - global_final_gain
    draws = int(config["analysis"]["bootstrap_draws"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    initial_harm = float(np.mean([row["metric_harm"] for row in initial_rows]))
    final_harm = float(np.mean([row["metric_harm"] for row in final_rows]))
    conditioner_change = float(
        (conditioner.projection.weight.detach() - initial_weight).norm()
    )
    summary = {
        "initial_online_loss_gain": float(np.mean([row["online_loss_gain"] for row in initial_rows])),
        "final_online_loss_gain": float(np.mean([row["online_loss_gain"] for row in final_rows])),
        "initial_metric_gain": float(initial_gain.mean()),
        "final_metric_gain": float(final_gain.mean()),
        "final_minus_initial_metric_gain": float(paired_gain.mean()),
        "exp053_learned_global_metric_gain": float(global_final_gain.mean()),
        "conditional_minus_learned_global_metric_gain": float(
            conditional_minus_global.mean()
        ),
        "initial_metric_harm_fraction": initial_harm,
        "final_metric_harm_fraction": final_harm,
        "conditioner_l2_change": conditioner_change,
        "initial_exp053_reproduction_max_abs": float(initial_reproduction_max_abs),
        "final_scale_mean": float(np.mean([row["scale_mean"] for row in final_rows])),
        "final_scale_std": float(np.mean([row["scale_std"] for row in final_rows])),
    }
    intervals = {
        "final_metric_gain_ci95": _ci(final_gain, draws, bootstrap_seed),
        "final_minus_initial_metric_gain_ci95": _ci(
            paired_gain, draws, bootstrap_seed + 1
        ),
        "conditional_minus_learned_global_metric_gain_ci95": _ci(
            conditional_minus_global, draws, bootstrap_seed + 2
        ),
    }
    checks = {
        "exact_coverage": len(trace) == len(fit_specs)
        and len(initial_rows) == len(audit_specs)
        and len(final_rows) == len(audit_specs),
        "finite": all(math.isfinite(value) for value in summary.values())
        and all(math.isfinite(row["conditioner_gradient_norm"]) for row in trace),
        "conditioner_changed": conditioner_change > 0,
        "zero_code_parity": max(
            row["zero_code_max_abs_difference"] for row in initial_rows + final_rows
        )
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "initial_reproduces_exp053": initial_reproduction_max_abs
        <= float(config["analysis"]["reproduction_tolerance"]),
        "positive_final_online_loss_gain": summary["final_online_loss_gain"] > 0,
        "positive_final_metric_gain_ci95": intervals["final_metric_gain_ci95"][0] > 0,
        "final_metric_gain_better_initial_ci95": intervals[
            "final_minus_initial_metric_gain_ci95"
        ][0]
        > 0,
        "final_metric_harm_below_limit": final_harm
        <= float(config["success"]["maximum_final_metric_harm_fraction"]),
        "final_harm_no_worse_initial": final_harm <= initial_harm,
        "beats_learned_global_positive_ci95": intervals[
            "conditional_minus_learned_global_metric_gain_ci95"
        ][0]
        > 0,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    passed = all(checks.values())
    checkpoint_hash = None
    if passed:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment": "EXP-055",
                "config": str(config_path),
                "conditioner_state_dict": conditioner.state_dict(),
                "initial_conditioner_state_dict": initial_state,
                "frozen_residual_state_dict": carrier.residual.state_dict(),
            },
            checkpoint_output,
        )
        checkpoint_hash = _sha256(checkpoint_output)
    result = {
        "experiment": "EXP-055",
        "stage": "conditioner_only_exact_meta_train_scene_audit",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "conditioner_parameters": sum(p.numel() for p in conditioner.parameters()),
        "complete_module_parameters": sum(p.numel() for p in conditioner.parameters())
        + sum(p.numel() for p in carrier.residual.parameters()),
        "summary": summary,
        "intervals": intervals,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "checkpoint_sha256": checkpoint_hash,
        "registered_gate": {"checks": checks, "passed": passed},
        "validation_accessed": False,
        "terminal_accessed": False,
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "training_trace": trace,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"initial_rows", "final_rows", "training_trace"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
