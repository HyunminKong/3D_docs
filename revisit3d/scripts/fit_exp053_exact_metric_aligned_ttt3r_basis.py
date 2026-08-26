#!/usr/bin/env python3
"""Fit and train-scene-audit the single EXP-053 TTT3R plasticity basis."""
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
    patch_center_points,
    symmetric_point_consistency,
)
from revisit3d.scripts.evaluate_exp052_ttt3r_metric_alignment_premise import (
    _model_view,
    _relative_point_loss,
    _rms_normalize,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _tuple_specs(sequences: list[str], targets: list[int], context: int) -> list[str]:
    return [
        sequence
        + " "
        + " ".join(
            f"{index:06d}" for index in range(target - context + 1, target + 1)
        )
        for sequence in sequences
        for target in targets
    ]


def _cache(carrier, gt_views: list[dict], patch_size: int) -> tuple[dict, dict, torch.Tensor]:
    state = None
    previous = current = auxiliary = None
    with torch.no_grad():
        for index, gt_view in enumerate(gt_views):
            prediction, state, current_auxiliary = carrier.step(
                _model_view(gt_view, index), state
            )
            if index == len(gt_views) - 2:
                previous = prediction
            if index == len(gt_views) - 1:
                current = prediction
                auxiliary = current_auxiliary
    assert previous is not None and current is not None and auxiliary is not None
    previous_points = patch_center_points(
        previous["pts3d_in_other_view"], patch_size
    ).detach()
    return current, auxiliary, previous_points


def _metric(prediction, gt_view, config) -> torch.Tensor:
    return _relative_point_loss(
        prediction["pts3d_in_self_view"],
        gt_view["depthmap"],
        gt_view["camera_intrinsics"],
        minimum_depth=float(config["metric"]["minimum_depth_m"]),
        maximum_depth=float(config["metric"]["maximum_depth_m"]),
    )


def _audit(carrier, dataset, specs, config) -> list[dict]:
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    rows = []
    for index, spec in enumerate(specs):
        gt_views = dataset[index]
        base, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
        tokens = auxiliary["decoder_patch_tokens"].shape[1]
        zero = torch.zeros(1, tokens, carrier.code_dim, device="cuda", requires_grad=True)
        zero_prediction = carrier.readout(auxiliary, code=zero)
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
            adapted = carrier.readout(auxiliary, code=code)
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
            }
        )
        del gt_views, base, auxiliary, previous_points, zero, zero_prediction
        del before_online, before_metric, gradient, code, adapted, after_online, after_metric
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _ci(values: np.ndarray, draws: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-053_exact_metric_aligned_ttt3r_basis_v10.yaml"
    )
    parser.add_argument("--confirm-train-only-fit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_only_fit or not torch.cuda.is_available():
        raise SystemExit("EXP-053 requires train-only fit confirmation and CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    checkpoint_output = Path(config["output"]["checkpoint"])
    preparation_path = Path(config["output"]["depth_preparation"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists() or checkpoint_output.exists():
        raise RuntimeError("EXP-053 artifact already exists")
    manifest = json.loads(manifest_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    fit_scenes = {value.split("/", 1)[0] for value in config["data"]["fit_sequences"]}
    audit_scenes = {value.split("/", 1)[0] for value in config["data"]["audit_sequences"]}
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and fit_scenes.isdisjoint(audit_scenes)
        and len(fit_scenes) == int(config["success"]["exact_fit_scenes"])
        and len(audit_scenes) == int(config["success"]["exact_audit_scenes"])
        and preparation["validation_accessed"] is False
        and preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-053 frozen contract failed")

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
        raise RuntimeError("EXP-053 anchor coverage differs from registration")
    dataset_args = {
        "split": "train",
        "ROOT": config["data"]["root"],
        "resolution": tuple(config["carrier"]["resolution"]),
        "seed": seed,
    }
    fit_dataset = SevenScenes(tuple_list=fit_specs, **dataset_args)
    audit_dataset = SevenScenes(tuple_list=audit_specs, **dataset_args)
    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
        update_type="ttt3r",
    ).cuda()
    carrier.eval()
    carrier.model.requires_grad_(False)
    carrier.residual.requires_grad_(True)
    initial_state = deepcopy(carrier.residual.state_dict())
    initial_weight = carrier.residual.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        carrier.residual.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    torch.cuda.reset_peak_memory_stats()
    initial_rows = _audit(carrier, audit_dataset, audit_specs, config)

    trace = []
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    for index, spec in enumerate(fit_specs):
        optimizer.zero_grad(set_to_none=True)
        gt_views = fit_dataset[index]
        _, auxiliary, previous_points = _cache(carrier, gt_views, patch_size)
        tokens = auxiliary["decoder_patch_tokens"].shape[1]
        zero = torch.zeros(1, tokens, carrier.code_dim, device="cuda", requires_grad=True)
        prediction = carrier.readout(auxiliary, code=zero)
        online_loss = symmetric_point_consistency(
            patch_center_points(prediction["pts3d_in_other_view"], patch_size),
            previous_points,
        )
        online_gradient = torch.autograd.grad(
            online_loss, zero, create_graph=True, retain_graph=True
        )[0]
        code = zero - step_size * _rms_normalize(online_gradient)
        adapted = carrier.readout(auxiliary, code=code)
        outer = _metric(adapted, gt_views[-1], config)
        outer.backward()
        gradient_norm = float(carrier.residual.projection.weight.grad.norm())
        optimizer.step()
        trace.append(
            {
                "step": index + 1,
                "sequence": spec.split()[0],
                "target_frame": int(spec.split()[-1]),
                "outer_metric": float(outer.detach()),
                "basis_gradient_norm": gradient_norm,
            }
        )
        print(json.dumps({"fit": index + 1, "total": len(fit_specs), **trace[-1]}), flush=True)
        del gt_views, auxiliary, previous_points, zero, prediction, online_loss
        del online_gradient, code, adapted, outer
        gc.collect()
        torch.cuda.empty_cache()

    final_rows = _audit(carrier, audit_dataset, audit_specs, config)
    initial_gain = np.asarray([row["metric_gain"] for row in initial_rows])
    final_gain = np.asarray([row["metric_gain"] for row in final_rows])
    paired_gain = final_gain - initial_gain
    draws = int(config["analysis"]["bootstrap_draws"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    initial_harm = float(np.mean([row["metric_harm"] for row in initial_rows]))
    final_harm = float(np.mean([row["metric_harm"] for row in final_rows]))
    basis_change = float(
        (carrier.residual.projection.weight.detach() - initial_weight).norm()
    )
    summary = {
        "initial_online_loss_gain": float(np.mean([row["online_loss_gain"] for row in initial_rows])),
        "final_online_loss_gain": float(np.mean([row["online_loss_gain"] for row in final_rows])),
        "initial_metric_gain": float(initial_gain.mean()),
        "final_metric_gain": float(final_gain.mean()),
        "final_minus_initial_metric_gain": float(paired_gain.mean()),
        "initial_metric_harm_fraction": initial_harm,
        "final_metric_harm_fraction": final_harm,
        "basis_l2_change": basis_change,
    }
    intervals = {
        "final_metric_gain_ci95": _ci(final_gain, draws, bootstrap_seed),
        "final_minus_initial_metric_gain_ci95": _ci(
            paired_gain, draws, bootstrap_seed + 1
        ),
    }
    checks = {
        "exact_coverage": len(trace) == len(fit_specs)
        and len(initial_rows) == len(audit_specs)
        and len(final_rows) == len(audit_specs),
        "finite": all(math.isfinite(value) for value in summary.values())
        and all(math.isfinite(row["basis_gradient_norm"]) for row in trace),
        "basis_changed": basis_change > 0,
        "zero_code_parity": max(
            row["zero_code_max_abs_difference"] for row in initial_rows + final_rows
        )
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "positive_final_online_loss_gain": summary["final_online_loss_gain"] > 0,
        "positive_final_metric_gain_ci95": intervals["final_metric_gain_ci95"][0] > 0,
        "final_metric_gain_better_initial_ci95": intervals[
            "final_minus_initial_metric_gain_ci95"
        ][0]
        > 0,
        "final_metric_harm_below_limit": final_harm
        <= float(config["success"]["maximum_final_metric_harm_fraction"]),
        "final_harm_no_worse_initial": final_harm <= initial_harm,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    passed = all(checks.values())
    checkpoint_hash = None
    if passed:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment": "EXP-053",
                "config": str(config_path),
                "residual_state_dict": carrier.residual.state_dict(),
                "initial_state_dict": initial_state,
            },
            checkpoint_output,
        )
        checkpoint_hash = _sha256(checkpoint_output)
    result = {
        "experiment": "EXP-053",
        "stage": "exact_metric_aligned_ttt3r_basis_train_scene_audit",
        "config": str(config_path),
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
    print(json.dumps({key: value for key, value in result.items() if key not in {"initial_rows", "final_rows", "training_trace"}}, indent=2))


if __name__ == "__main__":
    main()
