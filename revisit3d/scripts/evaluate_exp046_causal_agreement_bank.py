#!/usr/bin/env python3
"""Development evaluation of a causal agreement-addressed local-code bank."""
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

from revisit3d.backbones import FrozenCUT3RCarrier, transport_code_visual
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.evaluate_exp045_zero_agreement_validation import _bootstrap
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp042_learned_cut3r_plasticity_coordinate import (
    _loss_for_code,
    _online_code,
    _points,
    _prepare_pair,
    _scene_balanced,
)


def _agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean())


def _pooled(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float().mean(dim=1), dim=-1)


def _source_record(
    carrier: FrozenCUT3RCarrier,
    row: dict,
    load_images_for_eval,
    config: dict,
) -> dict:
    images = load_images_for_eval(
        [row["source_previous_rgb"], row["source_rgb"]],
        size=int(config["carrier"]["image_size"]),
        verbose=False,
        crop=bool(config["carrier"]["crop"]),
    )
    views = _views(images, [True, True])
    patch_size = int(config["plasticity"]["patch_size"])
    with torch.no_grad():
        previous, state, _ = carrier.step(views[0], None)
        previous_points = _points(previous, patch_size)
        base, _, auxiliary = carrier.step(views[1], state)
    source = _online_code(
        carrier,
        auxiliary,
        previous_points,
        patch_size=patch_size,
        normalized_step=float(config["plasticity"]["normalized_step"]),
    )
    parity = float((_points(base, patch_size) - source["base_points"]).abs().max())
    record = {
        "source_index": int(row["source_index"]),
        "pair_id": row["pair_id"],
        "code": source["code"].detach(),
        "features": auxiliary["image_tokens"].detach(),
        "pooled": _pooled(auxiliary["image_tokens"]).detach(),
        "cached_readout_parity_max_abs": parity,
    }
    del images, views
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-046_causal_agreement_bank_v10.yaml")
    parser.add_argument("--confirm-exposed-validation-development", action="store_true")
    args = parser.parse_args()
    if not args.confirm_exposed_validation_development:
        raise SystemExit("EXP-046 requires explicit exposed-validation confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-046 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-046 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    plasticity_checkpoint = Path(config["plasticity"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(plasticity_checkpoint)
        == config["plasticity"]["checkpoint_sha256"]
        and config["data"]["terminal_access"] is False
        and float(config["bank"]["application_threshold"]) == 0.0
        and config["bank"]["threshold_fitted"] is False
    ):
        raise RuntimeError("EXP-046 frozen development contract failed")
    manifest = json.loads(manifest_path.read_text())
    scenes = sorted({row["scene"] for row in manifest})
    if not (
        len(manifest) == int(config["data"]["exact_pairs"])
        and len(scenes) == int(config["data"]["exact_scenes"])
    ):
        raise RuntimeError("EXP-046 coverage contract failed")

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
    temperature = float(config["transport"]["visual_temperature"])
    threshold = float(config["bank"]["application_threshold"])
    results = []
    bank_sizes = {}
    query_index = 0

    for scene_index, scene in enumerate(scenes):
        scene_rows = [row for row in manifest if row["scene"] == scene]
        unique_source_rows = {}
        for row in scene_rows:
            unique_source_rows.setdefault(int(row["source_index"]), row)
        if len(unique_source_rows) > int(config["bank"]["capacity_per_scene"]):
            raise RuntimeError("EXP-046 bank capacity exceeded")
        bank = [
            _source_record(carrier, unique_source_rows[index], load_images_for_eval, config)
            for index in sorted(unique_source_rows)
        ]
        bank_sizes[scene] = len(bank)
        print(
            json.dumps(
                {"scene_bank": scene_index + 1, "scenes": len(scenes), "records": len(bank)}
            ),
            flush=True,
        )
        for row in sorted(scene_rows, key=lambda item: (item["target_index"], item["pair_id"])):
            eligible = [record for record in bank if record["source_index"] < int(row["target_index"])]
            if len(eligible) < int(config["success"]["minimum_candidates_per_query"]):
                raise RuntimeError("EXP-046 query has insufficient causal candidates")
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
                visual_temperature=temperature,
            )
            target = prepared["target"]
            target_pooled = _pooled(prepared["target_auxiliary"]["image_tokens"])
            transported_codes = []
            agreement_scores = []
            appearance_scores = []
            with torch.no_grad():
                for record in eligible:
                    transported, _ = transport_code_visual(
                        record["features"],
                        record["code"],
                        prepared["target_auxiliary"]["image_tokens"],
                        temperature=temperature,
                    )
                    transported_codes.append(transported)
                    agreement_scores.append(_agreement(transported, target["code"]))
                    appearance_scores.append(float((record["pooled"] * target_pooled).sum()))
            selections = {
                "agreement": int(np.argmax(agreement_scores)),
                "appearance": int(np.argmax(appearance_scores)),
            }
            generator = np.random.default_rng(int(config["bank"]["random_seed"]) + query_index)
            selections["random"] = int(generator.integers(0, len(eligible)))
            paired_matches = [
                index
                for index, record in enumerate(eligible)
                if record["source_index"] == int(row["source_index"])
            ]
            if len(paired_matches) != 1:
                raise RuntimeError("EXP-046 paired source is not uniquely present in causal bank")
            selections["paired"] = paired_matches[0]

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
                selected_losses = {}
                accepted = {}
                for name, selected in selections.items():
                    accepted[name] = agreement_scores[selected] > threshold
                    if accepted[name]:
                        selected_losses[name] = float(
                            _loss_for_code(
                                carrier,
                                prepared["target_auxiliary"],
                                target["code"] + transported_codes[selected],
                                prepared["target_previous_points"],
                                patch_size,
                            )
                        )
                    else:
                        selected_losses[name] = current_loss
            results.append(
                {
                    "pair_id": row["pair_id"],
                    "scene": scene,
                    "target_index": int(row["target_index"]),
                    "candidates": len(eligible),
                    "target_base_loss": target["base_loss"],
                    "target_current_loss": current_loss,
                    **{f"target_{name}_loss": selected_losses[name] for name in selections},
                    **{f"{name}_accepted": accepted[name] for name in selections},
                    **{
                        f"{name}_agreement": agreement_scores[selected]
                        for name, selected in selections.items()
                    },
                    "agreement_matches_paired": selections["agreement"] == selections["paired"],
                    "appearance_matches_paired": selections["appearance"] == selections["paired"],
                    "random_matches_paired": selections["random"] == selections["paired"],
                    "source_cached_readout_parity_max_abs": max(
                        record["cached_readout_parity_max_abs"] for record in eligible
                    ),
                    "target_cached_readout_parity_max_abs": prepared["parity"],
                }
            )
            query_index += 1
            print(json.dumps({"bank_query": query_index, "total": len(manifest)}), flush=True)
            del images, views, prepared, transported_codes
            gc.collect()
            torch.cuda.empty_cache()
        del bank
        gc.collect()
        torch.cuda.empty_cache()

    means = {
        key: _scene_balanced(results, key)
        for key in (
            "target_base_loss",
            "target_current_loss",
            "target_agreement_loss",
            "target_appearance_loss",
            "target_random_loss",
            "target_paired_loss",
            "candidates",
        )
    }
    comparisons = {
        "current_ttt_gain": ("target_base_loss", "target_current_loss"),
        "agreement_bank_gain_over_current": (
            "target_current_loss",
            "target_agreement_loss",
        ),
        "agreement_over_appearance": (
            "target_appearance_loss",
            "target_agreement_loss",
        ),
        "agreement_over_random": ("target_random_loss", "target_agreement_loss"),
        "paired_reference_gain_over_current": (
            "target_current_loss",
            "target_paired_loss",
        ),
    }
    uncertainty = _bootstrap(
        results,
        comparisons,
        draws=int(config["analysis"]["bootstrap_draws"]),
        seed=int(config["analysis"]["bootstrap_seed"]),
    )
    acceptance = {
        name: float(np.mean([row[f"{name}_accepted"] for row in results]))
        for name in ("agreement", "appearance", "random", "paired")
    }
    harm = {
        name: float(
            np.mean(
                [row[f"target_{name}_loss"] > row["target_current_loss"] for row in results]
            )
        )
        for name in ("agreement", "appearance", "random", "paired")
    }
    paired_match = {
        name: float(np.mean([row[f"{name}_matches_paired"] for row in results]))
        for name in ("agreement", "appearance", "random")
    }
    checks = {
        "exact_coverage": len(results) == int(config["data"]["exact_pairs"])
        and len({row["scene"] for row in results}) == int(config["data"]["exact_scenes"]),
        "minimum_candidates": min(row["candidates"] for row in results)
        >= int(config["success"]["minimum_candidates_per_query"]),
        "finite": all(math.isfinite(value) for value in means.values()),
        "exact_cached_readout_parity": max(
            max(
                row["source_cached_readout_parity_max_abs"],
                row["target_cached_readout_parity_max_abs"],
            )
            for row in results
        )
        == 0,
        "positive_agreement_bank_gain_ci95": uncertainty[
            "agreement_bank_gain_over_current"
        ]["ci95"][0]
        > 0,
        "agreement_better_appearance_ci95": uncertainty["agreement_over_appearance"][
            "ci95"
        ][0]
        > 0,
        "agreement_better_random_ci95": uncertainty["agreement_over_random"]["ci95"][0]
        > 0,
        "agreement_bank_harm_below_limit": harm["agreement"]
        <= float(config["success"]["maximum_agreement_bank_harm_fraction"]),
        "bank_capacity_respected": max(bank_sizes.values())
        <= int(config["success"]["maximum_bank_records_per_scene"]),
    }
    result = {
        "experiment": "EXP-046",
        "stage": "causal_parameter_free_bank_development",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "pairs": len(results),
        "scenes": len(scenes),
        "bank_sizes": bank_sizes,
        "means": means,
        "uncertainty": uncertainty,
        "acceptance_fractions": acceptance,
        "harm_fractions": harm,
        "paired_match_fractions": paired_match,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "selection_uses_pose": False,
        "selection_uses_pair_identity": False,
        "fitting_performed": False,
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
                "harm": harm,
                "paired_match": paired_match,
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
