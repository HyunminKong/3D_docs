#!/usr/bin/env python3
"""Evaluate unique memory headroom in metadata-defined low-parallax revisits."""
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
from revisit3d.scripts.evaluate_exp047_full_stream_bounded_bank import (
    _second_current_loss,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.fit_exp042_learned_cut3r_plasticity_coordinate import (
    _loss_for_code,
    _prepare_pair,
)


LOW = "low_parallax_complementary"
SUFFICIENT = "motion_sufficient"


def _agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean())


def _scene_values(rows: list[dict], regime: str, left: str, right: str) -> np.ndarray:
    scenes = sorted({row["scene"] for row in rows})
    values = []
    for scene in scenes:
        selected = [
            row for row in rows if row["scene"] == scene and row["information_regime"] == regime
        ]
        if len(selected) != 1:
            raise RuntimeError(f"EXP-049 expected one {regime} row in {scene}")
        values.append(selected[0][left] - selected[0][right])
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray, *, draws: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "positive_scenes": int((values > 0).sum()),
        "scenes": int(len(values)),
    }


def _uncertainty(rows: list[dict], config: dict) -> dict:
    draws = int(config["analysis"]["bootstrap_draws"])
    seed = int(config["analysis"]["bootstrap_seed"])
    low_second = _scene_values(
        rows, LOW, "target_current_loss", "target_second_current_loss"
    )
    low_oracle = _scene_values(
        rows, LOW, "target_second_current_loss", "target_future_oracle_loss"
    )
    sufficient_oracle = _scene_values(
        rows, SUFFICIENT, "target_second_current_loss", "target_future_oracle_loss"
    )
    return {
        "low_second_current_gain": _summary(low_second, draws=draws, seed=seed),
        "low_future_oracle_over_second_current": _summary(
            low_oracle, draws=draws, seed=seed + 1
        ),
        "sufficient_future_oracle_over_second_current": _summary(
            sufficient_oracle, draws=draws, seed=seed + 2
        ),
        "low_vs_sufficient_oracle_interaction": _summary(
            low_oracle - sufficient_oracle, draws=draws, seed=seed + 3
        ),
        "low_gated_memory_over_second_current": _summary(
            _scene_values(
                rows, LOW, "target_second_current_loss", "target_gated_memory_loss"
            ),
            draws=draws,
            seed=seed + 4,
        ),
        "low_raw_memory_over_shuffle": _summary(
            _scene_values(rows, LOW, "target_shuffle_loss", "target_raw_memory_loss"),
            draws=draws,
            seed=seed + 5,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-049_low_parallax_oracle_premise_v10.yaml"
    )
    parser.add_argument("--confirm-train-only-oracle", action="store_true")
    parser.add_argument("--smoke-pairs", type=int, default=0)
    args = parser.parse_args()
    if not args.confirm_train_only_oracle:
        raise SystemExit("EXP-049 requires explicit train-only oracle confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-049 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists() and args.smoke_pairs == 0:
        raise RuntimeError("EXP-049 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    plasticity_checkpoint = Path(config["plasticity"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(plasticity_checkpoint)
        == config["plasticity"]["checkpoint_sha256"]
        and config["data"]["role"] == "train"
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and float(config["routing"]["agreement_threshold"]) == 0.0
        and config["routing"]["threshold_fitted"] is False
        and config["oracle"]["online_deployable"] is False
    ):
        raise RuntimeError("EXP-049 frozen train-only contract failed")
    rows = json.loads(manifest_path.read_text())
    if not (
        len(rows) == int(config["data"]["exact_pairs"])
        and len({row["scene"] for row in rows}) == int(config["data"]["exact_scenes"])
        and all(row["role"] == "train" for row in rows)
        and {row["information_regime"] for row in rows} == {LOW, SUFFICIENT}
    ):
        raise RuntimeError("EXP-049 frozen manifest coverage failed")
    if args.smoke_pairs:
        rows = rows[: int(args.smoke_pairs)]

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
            raw_memory_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + transported,
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
            generator = torch.Generator(device="cpu").manual_seed(
                int(config["transport"]["spatial_shuffle_seed"]) + index
            )
            permutation = torch.randperm(
                transported.shape[1], generator=generator
            ).to(transported.device)
            shuffle_loss = float(
                _loss_for_code(
                    carrier,
                    prepared["target_auxiliary"],
                    target["code"] + transported[:, permutation],
                    prepared["target_previous_points"],
                    patch_size,
                )
            )
        second_current_loss = _second_current_loss(
            carrier,
            prepared["target_auxiliary"],
            target["code"],
            prepared["target_previous_points"],
            normalized_step=float(
                config["plasticity"]["second_current_normalized_step"]
            ),
            patch_size=patch_size,
        )
        agreement = _agreement(transported, target["code"])
        gated_loss = raw_memory_loss if agreement > 0.0 else current_loss
        future_oracle_loss = min(current_loss, raw_memory_loss)
        results.append(
            {
                "pair_id": row["pair_id"],
                "scene": row["scene"],
                "information_regime": row["information_regime"],
                "source_adjacent_translation_in_median_steps": row[
                    "source_adjacent_translation_in_median_steps"
                ],
                "target_adjacent_translation_in_median_steps": row[
                    "target_adjacent_translation_in_median_steps"
                ],
                "target_base_loss": target["base_loss"],
                "target_current_loss": current_loss,
                "target_second_current_loss": second_current_loss,
                "target_raw_memory_loss": raw_memory_loss,
                "target_gated_memory_loss": gated_loss,
                "target_future_oracle_loss": future_oracle_loss,
                "target_shuffle_loss": shuffle_loss,
                "memory_agreement": agreement,
                "memory_accepted": agreement > 0.0,
                "memory_future_useful": raw_memory_loss < current_loss,
                "cached_readout_parity_max_abs": prepared["parity"],
            }
        )
        print(
            json.dumps(
                {
                    "evaluated": index + 1,
                    "total": len(rows),
                    "regime": row["information_regime"],
                }
            ),
            flush=True,
        )
        del images, views, prepared
        gc.collect()
        torch.cuda.empty_cache()

    if args.smoke_pairs:
        print(json.dumps({"smoke_pairs": len(results), "passed": True}))
        return

    uncertainty = _uncertainty(results, config)
    means = {
        regime: {
            key: float(
                np.mean(
                    [row[key] for row in results if row["information_regime"] == regime]
                )
            )
            for key in (
                "target_base_loss",
                "target_current_loss",
                "target_second_current_loss",
                "target_raw_memory_loss",
                "target_gated_memory_loss",
                "target_future_oracle_loss",
                "target_shuffle_loss",
                "memory_agreement",
            )
        }
        for regime in (LOW, SUFFICIENT)
    }
    checks = {
        "exact_coverage": len(results) == int(config["data"]["exact_pairs"])
        and len({row["scene"] for row in results})
        == int(config["data"]["exact_scenes"])
        and all(
            sum(row["information_regime"] == regime for row in results)
            == int(config["data"]["exact_pairs_per_regime"])
            for regime in (LOW, SUFFICIENT)
        ),
        "finite": all(
            math.isfinite(value)
            for regime_means in means.values()
            for value in regime_means.values()
        ),
        "exact_cached_readout_parity": max(
            row["cached_readout_parity_max_abs"] for row in results
        )
        == 0,
        "positive_low_second_current_gain_ci95": uncertainty[
            "low_second_current_gain"
        ]["ci95"][0]
        > 0,
        "positive_low_oracle_over_second_current_ci95": uncertainty[
            "low_future_oracle_over_second_current"
        ]["ci95"][0]
        > 0,
        "positive_low_vs_sufficient_interaction_ci95": uncertainty[
            "low_vs_sufficient_oracle_interaction"
        ]["ci95"][0]
        > 0,
    }
    result = {
        "experiment": "EXP-049",
        "stage": config["purpose"],
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "pairs": len(results),
        "scenes": len({row["scene"] for row in results}),
        "means": means,
        "uncertainty": uncertainty,
        "acceptance_fraction": {
            regime: float(
                np.mean(
                    [
                        row["memory_accepted"]
                        for row in results
                        if row["information_regime"] == regime
                    ]
                )
            )
            for regime in (LOW, SUFFICIENT)
        },
        "future_useful_fraction": {
            regime: float(
                np.mean(
                    [
                        row["memory_future_useful"]
                        for row in results
                        if row["information_regime"] == regime
                    ]
                )
            )
            for regime in (LOW, SUFFICIENT)
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "fitting_performed": False,
        "pose_used_for_offline_regime_and_source": True,
        "pose_used_online": False,
        "future_oracle_used_online": False,
        "validation_accessed": False,
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
                "acceptance_fraction": result["acceptance_fraction"],
                "future_useful_fraction": result["future_useful_fraction"],
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
