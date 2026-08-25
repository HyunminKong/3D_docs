#!/usr/bin/env python3
"""Train-only oracle local-code reuse premise on frozen CUT3R."""
from __future__ import annotations

import argparse
import gc
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
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _adapt_code(carrier, view, state, previous_points, config):
    with torch.no_grad():
        base_prediction, next_state, auxiliary = carrier.step(view, state)
    tokens = auxiliary["decoder_patch_tokens"].shape[1]
    code = torch.zeros(1, tokens, carrier.code_dim, device="cuda", requires_grad=True)
    prediction, _, _ = carrier.step(view, state, code=code)
    points = patch_center_points(
        prediction["pts3d_in_other_view"], int(config["plasticity"]["patch_size"])
    )
    loss = symmetric_point_consistency(points, previous_points)
    gradient = torch.autograd.grad(loss, code, create_graph=False)[0]
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    updated_code = code.detach() - float(config["plasticity"]["normalized_step"]) * normalized.detach()
    with torch.no_grad():
        adapted_prediction, _, _ = carrier.step(view, state, code=updated_code)
        adapted_points = patch_center_points(
            adapted_prediction["pts3d_in_other_view"], int(config["plasticity"]["patch_size"])
        )
        adapted_loss = symmetric_point_consistency(adapted_points, previous_points)
    return {
        "base_prediction": base_prediction,
        "base_points": points.detach(),
        "base_loss": float(loss.detach()),
        "adapted_prediction": adapted_prediction,
        "adapted_points": adapted_points.detach(),
        "adapted_loss": float(adapted_loss),
        "code": updated_code,
        "features": auxiliary["image_tokens"],
        "gradient_norm": float(gradient.norm()),
        "next_state": next_state,
    }


def _selected(rows: list[dict], scenes: int, pairs: int) -> list[dict]:
    order = []
    for row in rows:
        if row["scene"] not in order:
            order.append(row["scene"])
        if len(order) == scenes:
            break
    selected = []
    for scene in order:
        selected.extend([row for row in rows if row["scene"] == scene][:pairs])
    return selected


def _scene_balanced(rows: list[dict], key: str) -> float:
    scenes = sorted({row["scene"] for row in rows})
    return float(np.mean([np.mean([row[key] for row in rows if row["scene"] == scene]) for scene in scenes]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-040_cut3r_oracle_reuse_premise_v10.yaml")
    parser.add_argument("--confirm-train-only-premise", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_only_premise:
        raise SystemExit("EXP-040 requires explicit train-only confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-040 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-040 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    checkpoint_path = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(checkpoint_path) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-040 source-safe contract failed")

    rows = json.loads(manifest_path.read_text())
    selected = _selected(
        rows,
        int(config["data"]["selected_scenes"]),
        int(config["data"]["pairs_per_scene"]),
    )
    if len(selected) != int(config["success"]["exact_pairs"]):
        raise RuntimeError("EXP-040 deterministic subset has wrong size")

    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    carrier = FrozenCUT3RCarrier(
        checkpoint_path,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        basis_seed=int(config["plasticity"]["basis_seed"]),
    ).cuda()
    carrier.eval()
    carrier.residual.requires_grad_(False)
    results = []

    for index, row in enumerate(selected):
        paths = [
            row["source_previous_rgb"],
            row["source_rgb"],
            row["target_previous_rgb"],
            row["target_rgb"],
        ]
        images = load_images_for_eval(
            paths,
            size=int(config["carrier"]["image_size"]),
            verbose=False,
            crop=bool(config["carrier"]["crop"]),
        )
        views = _views(images, [True, True, True, True])
        with torch.no_grad():
            source_previous_prediction, state, _ = carrier.step(views[0], None)
            source_previous_points = patch_center_points(
                source_previous_prediction["pts3d_in_other_view"],
                int(config["plasticity"]["patch_size"]),
            )
        source = _adapt_code(carrier, views[1], state, source_previous_points, config)

        with torch.no_grad():
            target_previous_prediction, target_state, _ = carrier.step(
                views[2], source["next_state"]
            )
            target_previous_points = patch_center_points(
                target_previous_prediction["pts3d_in_other_view"],
                int(config["plasticity"]["patch_size"]),
            )
        target = _adapt_code(carrier, views[3], target_state, target_previous_points, config)
        transported, transport_distance = transport_code_3d(
            source["base_points"], source["code"], target["base_points"]
        )
        full_code = target["code"] + transported
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["control"]["spatial_shuffle_seed"]) + index
        )
        permutation = torch.randperm(transported.shape[1], generator=generator).to("cuda")
        shuffled_code = target["code"] + transported[:, permutation]
        with torch.no_grad():
            full_prediction, _, _ = carrier.step(views[3], target_state, code=full_code)
            shuffled_prediction, _, _ = carrier.step(views[3], target_state, code=shuffled_code)
            full_points = patch_center_points(
                full_prediction["pts3d_in_other_view"], int(config["plasticity"]["patch_size"])
            )
            shuffled_points = patch_center_points(
                shuffled_prediction["pts3d_in_other_view"], int(config["plasticity"]["patch_size"])
            )
            full_loss = float(symmetric_point_consistency(full_points, target_previous_points))
            shuffled_loss = float(symmetric_point_consistency(shuffled_points, target_previous_points))

        results.append(
            {
                "pair_id": row["pair_id"],
                "scene": row["scene"],
                "source_base_loss": source["base_loss"],
                "source_ttt_loss": source["adapted_loss"],
                "target_base_loss": target["base_loss"],
                "target_current_loss": target["adapted_loss"],
                "target_full_loss": full_loss,
                "target_spatial_shuffle_loss": shuffled_loss,
                "source_gradient_norm": source["gradient_norm"],
                "target_gradient_norm": target["gradient_norm"],
                "mean_transport_distance": float(transport_distance.mean()),
            }
        )
        print(json.dumps({"evaluated": index + 1, "total": len(selected), "scene": row["scene"]}), flush=True)
        del images, views
        gc.collect()
        torch.cuda.empty_cache()

    means = {
        key: _scene_balanced(results, key)
        for key in (
            "source_base_loss",
            "source_ttt_loss",
            "target_base_loss",
            "target_current_loss",
            "target_full_loss",
            "target_spatial_shuffle_loss",
            "source_gradient_norm",
            "target_gradient_norm",
            "mean_transport_distance",
        )
    }
    gains = {
        "source_ttt": means["source_base_loss"] - means["source_ttt_loss"],
        "target_current_ttt": means["target_base_loss"] - means["target_current_loss"],
        "oracle_reuse_over_current": means["target_current_loss"] - means["target_full_loss"],
        "full_over_spatial_shuffle": means["target_spatial_shuffle_loss"] - means["target_full_loss"],
    }
    harm = float(np.mean([row["target_full_loss"] > row["target_current_loss"] for row in results]))
    checks = {
        "exact_coverage": len(results) == int(config["success"]["exact_pairs"])
        and len({row["scene"] for row in results}) == int(config["success"]["exact_scenes"]),
        "finite": all(np.isfinite(value) for value in means.values()),
        "positive_source_ttt_gain": gains["source_ttt"] > 0,
        "positive_target_current_ttt_gain": gains["target_current_ttt"] > 0,
        "positive_oracle_reuse_gain_over_current": gains["oracle_reuse_over_current"] > 0,
        "full_better_spatial_shuffle": gains["full_over_spatial_shuffle"] > 0,
        "reuse_harm_below_limit": harm <= float(config["success"]["maximum_reuse_harm_fraction"]),
    }
    result = {
        "experiment": "EXP-040",
        "stage": "cut3r_train_only_oracle_reuse_premise",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "carrier_frozen": True,
        "basis_fit_performed": False,
        "address_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "means": means,
        "gains": gains,
        "reuse_harm_fraction": harm,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "rows": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"means": means, "gains": gains, "harm": harm, "gate": result["registered_gate"]}, indent=2))


if __name__ == "__main__":
    main()
