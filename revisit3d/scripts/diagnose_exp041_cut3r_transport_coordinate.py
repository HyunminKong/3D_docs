#!/usr/bin/env python3
"""Decompose CUT3R update-coordinate versus transport failure on train data."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
    transport_code_3d,
    transport_code_visual,
)
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.evaluate_exp040_cut3r_oracle_reuse_premise import (
    _adapt_code,
    _scene_balanced,
    _selected,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-041_cut3r_transport_coordinate_diagnosis_v10.yaml")
    parser.add_argument("--confirm-train-only-diagnosis", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_only_diagnosis:
        raise SystemExit("EXP-041 requires explicit train-only confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-041 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-041 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    checkpoint_path = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(checkpoint_path) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-041 source-safe contract failed")

    rows = json.loads(manifest_path.read_text())
    selected = _selected(rows, int(config["data"]["selected_scenes"]), int(config["data"]["pairs_per_scene"]))
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
        images = load_images_for_eval(
            [row[key] for key in ("source_previous_rgb", "source_rgb", "target_previous_rgb", "target_rgb")],
            size=int(config["carrier"]["image_size"]),
            verbose=False,
            crop=bool(config["carrier"]["crop"]),
        )
        views = _views(images, [True, True, True, True])
        patch_size = int(config["plasticity"]["patch_size"])
        with torch.no_grad():
            source_previous, state, _ = carrier.step(views[0], None)
            source_previous_points = patch_center_points(source_previous["pts3d_in_other_view"], patch_size)
        source = _adapt_code(carrier, views[1], state, source_previous_points, config)
        with torch.no_grad():
            target_previous, target_state, _ = carrier.step(views[2], source["next_state"])
            target_previous_points = patch_center_points(target_previous["pts3d_in_other_view"], patch_size)
        target = _adapt_code(carrier, views[3], target_state, target_previous_points, config)

        geometry, geometry_distance = transport_code_3d(
            source["base_points"], source["code"], target["base_points"]
        )
        visual, visual_peak = transport_code_visual(
            source["features"],
            source["code"],
            target["features"],
            temperature=float(config["transport"]["visual_temperature"]),
        )
        candidates = {
            "untransported": source["code"],
            "visual": visual,
            "geometry": geometry,
        }
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["transport"]["spatial_shuffle_seed"]) + index
        )
        permutation = torch.randperm(visual.shape[1], generator=generator).to("cuda")
        candidates["visual_shuffle"] = visual[:, permutation]
        losses, alignments = {}, {}
        with torch.no_grad():
            for name, past_code in candidates.items():
                prediction, _, _ = carrier.step(
                    views[3], target_state, code=target["code"] + past_code
                )
                points = patch_center_points(prediction["pts3d_in_other_view"], patch_size)
                losses[name] = float(symmetric_point_consistency(points, target_previous_points))
                alignments[name] = _alignment(past_code, target["code"])
        results.append(
            {
                "pair_id": row["pair_id"],
                "scene": row["scene"],
                "current_loss": target["adapted_loss"],
                "losses": losses,
                "alignments": alignments,
                "geometry_mean_distance": float(geometry_distance.mean()),
                "visual_mean_peak_weight": float(visual_peak.mean()),
            }
        )
        print(json.dumps({"evaluated": index + 1, "total": len(selected)}), flush=True)
        del images, views
        gc.collect()
        torch.cuda.empty_cache()

    names = ("untransported", "visual", "geometry", "visual_shuffle")
    expanded = [
        {
            "scene": row["scene"],
            "current": row["current_loss"],
            **{f"loss_{name}": row["losses"][name] for name in names},
            **{f"alignment_{name}": row["alignments"][name] for name in names},
        }
        for row in results
    ]
    means = {"current": _scene_balanced(expanded, "current")}
    for name in names:
        means[f"loss_{name}"] = _scene_balanced(expanded, f"loss_{name}")
        means[f"alignment_{name}"] = _scene_balanced(expanded, f"alignment_{name}")
    gains = {name: means["current"] - means[f"loss_{name}"] for name in names}
    harms = {
        name: float(np.mean([row["losses"][name] > row["current_loss"] for row in results]))
        for name in names
    }
    eligible = {
        name: gains[name] > 0
        and means[f"alignment_{name}"] > 0
        and (name == "visual_shuffle" or means[f"loss_{name}"] < means["loss_visual_shuffle"])
        for name in ("untransported", "visual", "geometry")
    }
    accepted = [name for name in ("untransported", "visual", "geometry") if eligible[name]]
    decision = min(accepted, key=lambda name: means[f"loss_{name}"]) if accepted else "no_raw_carrier"
    result = {
        "experiment": "EXP-041",
        "stage": "cut3r_train_only_transport_coordinate_diagnosis",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "pairs": len(results),
        "scenes": len({row["scene"] for row in results}),
        "means": means,
        "gains_over_current": gains,
        "harm_fractions": harms,
        "eligible_carriers": eligible,
        "registered_decision": decision,
        "basis_fit_performed": False,
        "address_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("means", "gains_over_current", "harm_fractions", "eligible_carriers", "registered_decision")}, indent=2))


if __name__ == "__main__":
    main()
