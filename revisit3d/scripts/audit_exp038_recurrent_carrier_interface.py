#!/usr/bin/env python3
"""Zero-fit CUT3R local-plasticity interface and differentiability audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
    transport_code_3d,
)
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _build_sequence, _views
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


POINT_KEYS = ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose", "conf", "conf_self")


def _maximum_error(left: dict, right: dict) -> float:
    values = []
    for key in POINT_KEYS:
        if key in left and key in right:
            values.append(float((left[key].detach().cpu() - right[key].detach().cpu()).abs().max()))
    if not values:
        raise RuntimeError("no shared geometry outputs for parity audit")
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-038_recurrent_carrier_interface_v10.yaml")
    parser.add_argument("--confirm-interface-audit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_interface_audit:
        raise SystemExit("EXP-038 requires explicit interface-audit confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-038 requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-038 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    checkpoint_path = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(checkpoint_path) == config["carrier"]["checkpoint_sha256"]
        and config["carrier"]["mode"] == "cut3r"
    ):
        raise RuntimeError("EXP-038 frozen input contract failed")

    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    events = json.loads(manifest_path.read_text())
    sequence = config["data"]["probe_sequence"]
    paths, updates, _ = _build_sequence(events, sequence)
    count = int(config["data"]["probe_frames"])
    paths, updates = paths[:count], updates[:count]
    images = load_images_for_eval(
        paths,
        size=int(config["carrier"]["image_size"]),
        verbose=False,
        crop=bool(config["carrier"]["crop"]),
    )
    views = _views(images, updates)

    carrier = FrozenCUT3RCarrier(
        checkpoint_path,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
    ).cuda()
    carrier.eval()
    carrier.residual.requires_grad_(False)

    with torch.no_grad():
        native, _ = carrier.model.forward_recurrent_lighter(views, device="cuda", ret_state=False)
        state = None
        custom = []
        states_in = []
        auxiliaries = []
        for view in views:
            states_in.append(state)
            prediction, state, auxiliary = carrier.step(view, state)
            custom.append({key: value.detach().cpu() for key, value in prediction.items()})
            auxiliaries.append(auxiliary)
    native_errors = [_maximum_error(left, right) for left, right in zip(native, custom)]
    native_parity_error = max(native_errors)

    probe_index = 1
    incoming_state = states_in[probe_index]
    if incoming_state is None:
        raise RuntimeError("adaptation probe requires an initialized recurrent state")
    patch_tokens = auxiliaries[probe_index]["decoder_patch_tokens"].shape[1]
    code = torch.zeros(
        1,
        patch_tokens,
        int(config["plasticity"]["code_dim"]),
        device="cuda",
        requires_grad=True,
    )
    zero_prediction, _, _ = carrier.step(views[probe_index], incoming_state, code=code)
    zero_code_error = _maximum_error(zero_prediction, custom[probe_index])

    patch_size = int(config["plasticity"]["patch_size"])
    previous_points = patch_center_points(
        custom[probe_index - 1]["pts3d_in_other_view"].cuda(), patch_size
    ).detach()
    current_points = patch_center_points(
        zero_prediction["pts3d_in_other_view"], patch_size
    )
    loss_before = symmetric_point_consistency(current_points, previous_points)
    gradient = torch.autograd.grad(loss_before, code, create_graph=False)[0]
    gradient_norm = float(gradient.norm())
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    step_size = float(config["plasticity"]["diagnostic_normalized_step"])
    updated_code = (code.detach() - step_size * normalized.detach())
    updated_prediction, _, _ = carrier.step(
        views[probe_index], incoming_state, code=updated_code
    )
    updated_points = patch_center_points(
        updated_prediction["pts3d_in_other_view"], patch_size
    )
    loss_after = symmetric_point_consistency(updated_points, previous_points)
    nonzero_output_change = _maximum_error(updated_prediction, zero_prediction)

    identity, identity_distance = transport_code_3d(
        current_points.detach(), updated_code, current_points.detach()
    )
    identity_rmse = float((identity - updated_code).square().mean().sqrt())
    previous_code = torch.linspace(
        -1,
        1,
        previous_points.shape[1],
        device="cuda",
    )[None, :, None].expand(-1, -1, updated_code.shape[-1])
    cross_transport, cross_distance = transport_code_3d(
        previous_points, previous_code, current_points.detach()
    )

    checks = {
        "native_cut3r_parity": native_parity_error
        <= float(config["success"]["maximum_native_parity_error"]),
        "zero_code_parity": zero_code_error
        <= float(config["success"]["maximum_zero_code_error"]),
        "finite_nonzero_code_gradient": np.isfinite(gradient_norm)
        and gradient_norm >= float(config["success"]["minimum_gradient_norm"]),
        "diagnostic_online_loss_decreased": float(loss_after) < float(loss_before),
        "nonzero_code_changes_geometry": np.isfinite(nonzero_output_change)
        and nonzero_output_change > 0,
        "identity_3d_transport": identity_rmse
        <= float(config["success"]["maximum_identity_transport_rmse"]),
        "cross_view_3d_transport_finite": bool(torch.isfinite(cross_transport).all())
        and bool(torch.isfinite(cross_distance).all()),
    }
    result = {
        "experiment": "EXP-038",
        "stage": "recurrent_carrier_interface_audit",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "purpose": config["purpose"],
        "probe": {"sequence": sequence, "frames": count, "query_or_depth_accessed": False},
        "carrier": {
            "mode": "cut3r",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "frozen": True,
        },
        "plasticity": {
            "code_dim": carrier.code_dim,
            "patch_tokens": patch_tokens,
            "code_bytes_float32": int(updated_code.numel() * updated_code.element_size()),
            "trainable_basis_parameters": sum(p.numel() for p in carrier.residual.parameters()),
        },
        "measurements": {
            "native_per_frame_max_abs_error": native_errors,
            "native_max_abs_error": native_parity_error,
            "zero_code_max_abs_error": zero_code_error,
            "online_loss_before": float(loss_before),
            "online_loss_after": float(loss_after),
            "code_gradient_norm": gradient_norm,
            "nonzero_output_max_change": nonzero_output_change,
            "identity_transport_rmse": identity_rmse,
            "identity_transport_max_distance": float(identity_distance.max()),
            "cross_transport_mean_distance": float(cross_distance.mean()),
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "model_fit_performed": False,
        "final_test_evidence": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
