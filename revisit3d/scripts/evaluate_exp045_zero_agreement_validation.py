#!/usr/bin/env python3
"""One-shot validation of frozen zero-agreement memory routing."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp042_learned_cut3r_plasticity_coordinate import (
    _loss_for_code,
    _prepare_pair,
    _scene_balanced,
)


def _agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean())


def _bootstrap(
    rows: list[dict], comparisons: dict[str, tuple[str, str]], *, draws: int, seed: int
) -> dict:
    scenes = sorted({row["scene"] for row in rows})
    result = {}
    for offset, (name, (left, right)) in enumerate(comparisons.items()):
        values = np.asarray(
            [
                np.mean(
                    [row[left] - row[right] for row in rows if row["scene"] == scene]
                )
                for scene in scenes
            ],
            dtype=np.float64,
        )
        generator = np.random.default_rng(seed + offset)
        indices = generator.integers(0, len(values), size=(draws, len(values)))
        samples = values[indices].mean(axis=1)
        result[name] = {
            "mean": float(values.mean()),
            "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            "positive_scenes": int((values > 0).sum()),
            "scenes": len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-045_zero_agreement_validation_v10.yaml"
    )
    parser.add_argument("--confirm-one-shot-validation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_one_shot_validation:
        raise SystemExit("EXP-045 requires explicit one-shot validation confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-045 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-045 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    plasticity_checkpoint = Path(config["plasticity"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(plasticity_checkpoint)
        == config["plasticity"]["checkpoint_sha256"]
        and config["data"]["terminal_access"] is False
        and float(config["routing"]["threshold"]) == 0.0
        and config["routing"]["threshold_fitted"] is False
    ):
        raise RuntimeError("EXP-045 frozen validation contract failed")
    rows = json.loads(manifest_path.read_text())
    if not (
        len(rows) == int(config["data"]["exact_pairs"])
        and len({row["scene"] for row in rows}) == int(config["data"]["exact_scenes"])
        and all(row["role"] == "validation" for row in rows)
    ):
        raise RuntimeError("EXP-045 validation coverage contract failed")

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
    ).cuda()
    payload = torch.load(plasticity_checkpoint, map_location="cpu")
    carrier.residual.load_state_dict(payload["residual_state_dict"])
    carrier.eval()
    carrier.residual.requires_grad_(False)
    patch_size = int(config["plasticity"]["patch_size"])
    threshold = float(config["routing"]["threshold"])
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
            normalized_step=float(config["plasticity"]["normalized_step"]),
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
        shuffled = transported[:, permutation]
        correct_agreement = _agreement(transported, target["code"])
        shuffle_agreement = _agreement(shuffled, target["code"])
        correct_accepted = correct_agreement > threshold
        shuffle_accepted = shuffle_agreement > threshold
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
            ungated_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + transported,
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
            shuffle_full_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + shuffled,
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
                "target_ungated_loss": ungated_loss,
                "target_gated_loss": ungated_loss if correct_accepted else current_loss,
                "target_shuffle_full_loss": shuffle_full_loss,
                "target_gated_shuffle_loss": shuffle_full_loss
                if shuffle_accepted
                else current_loss,
                "correct_agreement": correct_agreement,
                "shuffle_agreement": shuffle_agreement,
                "correct_accepted": correct_accepted,
                "shuffle_accepted": shuffle_accepted,
                "cached_readout_parity_max_abs": prepared["parity"],
            }
        )
        print(json.dumps({"validated": index + 1, "total": len(rows)}), flush=True)
        del images, views, prepared
        gc.collect()
        torch.cuda.empty_cache()

    mean_keys = (
        "target_base_loss",
        "target_current_loss",
        "target_ungated_loss",
        "target_gated_loss",
        "target_shuffle_full_loss",
        "target_gated_shuffle_loss",
        "correct_agreement",
        "shuffle_agreement",
        "cached_readout_parity_max_abs",
    )
    means = {key: _scene_balanced(results, key) for key in mean_keys}
    comparisons = {
        "current_ttt_gain": ("target_base_loss", "target_current_loss"),
        "gated_reuse_gain_over_current": ("target_current_loss", "target_gated_loss"),
        "gated_over_ungated": ("target_ungated_loss", "target_gated_loss"),
        "gated_correct_over_gated_shuffle": (
            "target_gated_shuffle_loss",
            "target_gated_loss",
        ),
    }
    uncertainty = _bootstrap(
        results,
        comparisons,
        draws=int(config["analysis"]["bootstrap_draws"]),
        seed=int(config["analysis"]["bootstrap_seed"]),
    )
    acceptance = float(np.mean([row["correct_accepted"] for row in results]))
    shuffle_acceptance = float(np.mean([row["shuffle_accepted"] for row in results]))
    harm = float(
        np.mean(
            [row["target_gated_loss"] > row["target_current_loss"] for row in results]
        )
    )
    checks = {
        "exact_coverage": len(results) == int(config["data"]["exact_pairs"])
        and len({row["scene"] for row in results}) == int(config["data"]["exact_scenes"]),
        "finite": all(math.isfinite(value) for value in means.values())
        and all(
            math.isfinite(bound)
            for summary in uncertainty.values()
            for bound in summary["ci95"]
        ),
        "exact_cached_readout_parity": max(
            row["cached_readout_parity_max_abs"] for row in results
        )
        == 0,
        "positive_current_ttt_gain_ci95": uncertainty["current_ttt_gain"]["ci95"][0]
        > 0,
        "positive_gated_reuse_gain_ci95": uncertainty[
            "gated_reuse_gain_over_current"
        ]["ci95"][0]
        > 0,
        "gated_better_ungated_ci95": uncertainty["gated_over_ungated"]["ci95"][0]
        > 0,
        "gated_correct_better_gated_shuffle_ci95": uncertainty[
            "gated_correct_over_gated_shuffle"
        ]["ci95"][0]
        > 0,
        "gated_harm_below_limit": harm
        <= float(config["success"]["maximum_gated_harm_fraction"]),
        "nondegenerate_acceptance": float(config["success"]["minimum_acceptance_fraction"])
        <= acceptance
        <= float(config["success"]["maximum_acceptance_fraction"]),
    }
    result = {
        "experiment": "EXP-045",
        "stage": "frozen_zero_agreement_validation",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "pairs": len(results),
        "scenes": len({row["scene"] for row in results}),
        "means": means,
        "uncertainty": uncertainty,
        "acceptance_fraction": acceptance,
        "shuffle_acceptance_fraction": shuffle_acceptance,
        "gated_harm_fraction": harm,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "fitting_performed": False,
        "validation_accessed": True,
        "terminal_accessed": False,
        "rows": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "means": means,
                "uncertainty": uncertainty,
                "acceptance": acceptance,
                "shuffle_acceptance": shuffle_acceptance,
                "harm": harm,
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
