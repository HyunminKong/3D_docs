#!/usr/bin/env python3
"""Run continuous CUT3R streams with bounded every-frame adaptation banks."""
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
    _scene_balanced,
)


def _agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(1), right.flatten(1), dim=-1).mean())


def _pooled(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float().mean(dim=1), dim=-1)


def _address(
    bank: list[dict],
    target_features: torch.Tensor,
    target_code: torch.Tensor,
    *,
    temperature: float,
) -> dict:
    transported_codes = []
    agreements = []
    appearances = []
    target_pooled = _pooled(target_features)
    with torch.no_grad():
        for record in bank:
            transported, _ = transport_code_visual(
                record["features"],
                record["code"],
                target_features,
                temperature=temperature,
            )
            transported_codes.append(transported)
            agreements.append(_agreement(transported, target_code))
            appearances.append(float((record["pooled"] * target_pooled).sum()))
    return {
        "transported": transported_codes,
        "agreements": agreements,
        "appearances": appearances,
    }


def _selected_loss(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    current_code: torch.Tensor,
    previous_points: torch.Tensor,
    transported: torch.Tensor,
    agreement: float,
    *,
    threshold: float,
    patch_size: int,
    current_loss: float,
) -> tuple[float, bool]:
    accepted = agreement > threshold
    if not accepted:
        return current_loss, False
    with torch.no_grad():
        loss = float(
            _loss_for_code(
                carrier,
                auxiliary,
                current_code + transported,
                previous_points,
                patch_size,
            )
        )
    return loss, True


def _second_current_loss(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    current_code: torch.Tensor,
    previous_points: torch.Tensor,
    *,
    normalized_step: float,
    patch_size: int,
) -> float:
    """Apply an equal-size second TTT step using only the current observation."""
    code = current_code.detach().clone().requires_grad_(True)
    prediction = carrier.readout(auxiliary, code=code)
    points = _points(prediction, patch_size)
    loss = torch.linalg.vector_norm(
        points[:, :, None, :] - previous_points[:, None, :, :], dim=-1
    )
    loss = 0.5 * (loss.min(dim=-1).values.mean() + loss.min(dim=-2).values.mean())
    gradient = torch.autograd.grad(loss, code, create_graph=False)[0]
    normalized = gradient / gradient.square().mean().sqrt().clamp_min(1e-12)
    updated = code.detach() - float(normalized_step) * normalized.detach()
    with torch.no_grad():
        return float(
            _loss_for_code(
                carrier, auxiliary, updated, previous_points, patch_size
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-047_full_stream_bounded_bank_v10.yaml")
    parser.add_argument("--confirm-full-stream-development", action="store_true")
    args = parser.parse_args()
    if not args.confirm_full_stream_development:
        raise SystemExit("EXP-047 requires explicit full-stream development confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-047 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-047 result already exists")
    manifest_path = Path(config["data"]["query_manifest"])
    carrier_checkpoint = Path(config["carrier"]["checkpoint"])
    plasticity_checkpoint = Path(config["plasticity"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["query_manifest_sha256"]
        and _sha256(carrier_checkpoint) == config["carrier"]["checkpoint_sha256"]
        and _sha256(plasticity_checkpoint)
        == config["plasticity"]["checkpoint_sha256"]
        and config["data"]["terminal_access"] is False
        and config["bank"]["write_policy"] == "every_frame_after_prediction"
        and float(config["bank"]["application_threshold"]) == 0.0
        and config["bank"]["threshold_fitted"] is False
    ):
        raise RuntimeError("EXP-047 frozen stream contract failed")
    queries = json.loads(manifest_path.read_text())
    scenes = sorted({row["scene"] for row in queries})
    if not (
        len(queries) == int(config["data"]["exact_queries"])
        and len(scenes) == int(config["data"]["exact_scenes"])
    ):
        raise RuntimeError("EXP-047 query coverage contract failed")

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
    capacity = int(config["bank"]["capacity"])
    patch_size = int(config["plasticity"]["patch_size"])
    temperature = float(config["transport"]["visual_temperature"])
    threshold = float(config["bank"]["application_threshold"])
    compute_second_current = bool(config["controls"].get("second_current_step", False))
    results = []
    processed_frames = 0
    maximum_bank_sizes = {"reservoir": 0, "fifo": 0}

    for scene_number, scene in enumerate(scenes):
        scene_queries = [row for row in queries if row["scene"] == scene]
        query_by_path = {str(Path(row["target_rgb"]).resolve()): row for row in scene_queries}
        if len(query_by_path) != len(scene_queries):
            raise RuntimeError("EXP-047 query paths are not unique")
        image_directory = Path(scene_queries[0]["target_rgb"]).parent
        all_frames = sorted(image_directory.glob("frame_*.png"))
        last_index = max(int(row["target_index"]) for row in scene_queries)
        frames = all_frames[: last_index + 1]
        if len(frames) != last_index + 1:
            raise RuntimeError("EXP-047 stream frames are incomplete")
        for row in scene_queries:
            if all_frames[int(row["target_index"])].resolve() != Path(row["target_rgb"]).resolve():
                raise RuntimeError("EXP-047 frame-index/path mapping failed")

        reservoir: list[dict] = []
        fifo: list[dict] = []
        reservoir_seen = 0
        reservoir_generator = np.random.default_rng(
            int(config["bank"]["reservoir_seed"]) + scene_number
        )
        state = None
        previous_points = None
        for frame_index, frame_path in enumerate(frames):
            images = load_images_for_eval(
                [str(frame_path)],
                size=int(config["carrier"]["image_size"]),
                verbose=False,
                crop=bool(config["carrier"]["crop"]),
            )
            view = _views(images, [True])[0]
            with torch.no_grad():
                base_prediction, next_state, auxiliary = carrier.step(view, state)
                base_points = _points(base_prediction, patch_size)
            processed_frames += 1
            if previous_points is not None:
                current = _online_code(
                    carrier,
                    auxiliary,
                    previous_points,
                    patch_size=patch_size,
                    normalized_step=float(config["plasticity"]["normalized_step"]),
                )
                parity = float((base_points - current["base_points"]).abs().max())
                query = query_by_path.get(str(frame_path.resolve()))
                if query is not None:
                    if not reservoir or not fifo:
                        raise RuntimeError("EXP-047 query encountered an empty bank")
                    with torch.no_grad():
                        current_loss = float(
                            _loss_for_code(
                                carrier,
                                auxiliary,
                                current["code"],
                                previous_points,
                                patch_size,
                            )
                        )
                    second_current_loss = None
                    if compute_second_current:
                        second_current_loss = _second_current_loss(
                            carrier,
                            auxiliary,
                            current["code"],
                            previous_points,
                            normalized_step=float(
                                config["controls"]["second_current_normalized_step"]
                            ),
                            patch_size=patch_size,
                        )
                    reservoir_address = _address(
                        reservoir,
                        auxiliary["image_tokens"],
                        current["code"],
                        temperature=temperature,
                    )
                    fifo_address = _address(
                        fifo,
                        auxiliary["image_tokens"],
                        current["code"],
                        temperature=temperature,
                    )
                    agreement_index = int(np.argmax(reservoir_address["agreements"]))
                    appearance_index = int(np.argmax(reservoir_address["appearances"]))
                    random_generator = np.random.default_rng(
                        int(config["bank"]["random_address_seed"]) + len(results)
                    )
                    random_index = int(random_generator.integers(0, len(reservoir)))
                    fifo_index = int(np.argmax(fifo_address["agreements"]))
                    selections = {
                        "reservoir_agreement": (
                            reservoir,
                            reservoir_address,
                            agreement_index,
                        ),
                        "reservoir_appearance": (
                            reservoir,
                            reservoir_address,
                            appearance_index,
                        ),
                        "reservoir_random": (
                            reservoir,
                            reservoir_address,
                            random_index,
                        ),
                        "fifo_agreement": (fifo, fifo_address, fifo_index),
                    }
                    selected_losses = {}
                    accepted = {}
                    selected_ages = {}
                    selected_agreements = {}
                    for name, (bank, address, selected) in selections.items():
                        selected_losses[name], accepted[name] = _selected_loss(
                            carrier,
                            auxiliary,
                            current["code"],
                            previous_points,
                            address["transported"][selected],
                            address["agreements"][selected],
                            threshold=threshold,
                            patch_size=patch_size,
                            current_loss=current_loss,
                        )
                        selected_ages[name] = frame_index - bank[selected]["frame_index"]
                        selected_agreements[name] = address["agreements"][selected]
                    results.append(
                        {
                            "pair_id": query["pair_id"],
                            "scene": scene,
                            "frame_index": frame_index,
                            "target_base_loss": current["base_loss"],
                            "target_current_loss": current_loss,
                            **(
                                {"target_second_current_loss": second_current_loss}
                                if compute_second_current
                                else {}
                            ),
                            **{
                                f"target_{name}_loss": loss
                                for name, loss in selected_losses.items()
                            },
                            **{f"{name}_accepted": value for name, value in accepted.items()},
                            **{
                                f"{name}_selected_age": age
                                for name, age in selected_ages.items()
                            },
                            **{
                                f"{name}_agreement": value
                                for name, value in selected_agreements.items()
                            },
                            "reservoir_size": len(reservoir),
                            "fifo_size": len(fifo),
                            "cached_readout_parity_max_abs": parity,
                        }
                    )

                # Predict/retrieve/apply precedes this write.
                record = {
                    "frame_index": frame_index,
                    "code": current["code"].detach(),
                    "features": auxiliary["image_tokens"].detach(),
                    "pooled": _pooled(auxiliary["image_tokens"]).detach(),
                }
                reservoir_seen += 1
                if len(reservoir) < capacity:
                    reservoir.append(record)
                else:
                    replacement = int(reservoir_generator.integers(0, reservoir_seen))
                    if replacement < capacity:
                        reservoir[replacement] = record
                fifo.append(record)
                if len(fifo) > capacity:
                    fifo.pop(0)
                maximum_bank_sizes["reservoir"] = max(
                    maximum_bank_sizes["reservoir"], len(reservoir)
                )
                maximum_bank_sizes["fifo"] = max(maximum_bank_sizes["fifo"], len(fifo))
            previous_points = base_points.detach()
            state = next_state
            if processed_frames % 50 == 0 or frame_index == len(frames) - 1:
                print(
                    json.dumps(
                        {
                            "scene": scene_number + 1,
                            "scenes": len(scenes),
                            "scene_frame": frame_index + 1,
                            "scene_frames": len(frames),
                            "processed_frames": processed_frames,
                            "queries": len(results),
                        }
                    ),
                    flush=True,
                )
            del images, view, base_prediction, auxiliary
            if processed_frames % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        if len([row for row in results if row["scene"] == scene]) != len(scene_queries):
            raise RuntimeError("EXP-047 did not evaluate every scene query")
        del reservoir, fifo, state, previous_points
        gc.collect()
        torch.cuda.empty_cache()

    policy_names = (
        "reservoir_agreement",
        "reservoir_appearance",
        "reservoir_random",
        "fifo_agreement",
    )
    means = {
        key: _scene_balanced(results, key)
        for key in (
            "target_base_loss",
            "target_current_loss",
            *(
                ("target_second_current_loss",)
                if compute_second_current
                else ()
            ),
            *(f"target_{name}_loss" for name in policy_names),
        )
    }
    if compute_second_current:
        comparisons = {
            "current_ttt_gain": ("target_base_loss", "target_current_loss"),
            "second_current_gain_over_current": (
                "target_current_loss",
                "target_second_current_loss",
            ),
            "fifo_agreement_gain_over_current": (
                "target_current_loss",
                "target_fifo_agreement_loss",
            ),
            "fifo_agreement_over_second_current": (
                "target_second_current_loss",
                "target_fifo_agreement_loss",
            ),
            "fifo_agreement_over_reservoir_appearance": (
                "target_reservoir_appearance_loss",
                "target_fifo_agreement_loss",
            ),
            "fifo_agreement_over_reservoir_random": (
                "target_reservoir_random_loss",
                "target_fifo_agreement_loss",
            ),
        }
    else:
        comparisons = {
            "current_ttt_gain": ("target_base_loss", "target_current_loss"),
            "reservoir_agreement_gain_over_current": (
                "target_current_loss",
                "target_reservoir_agreement_loss",
            ),
            "reservoir_agreement_over_appearance": (
                "target_reservoir_appearance_loss",
                "target_reservoir_agreement_loss",
            ),
            "reservoir_agreement_over_random": (
                "target_reservoir_random_loss",
                "target_reservoir_agreement_loss",
            ),
            "reservoir_agreement_over_fifo": (
                "target_fifo_agreement_loss",
                "target_reservoir_agreement_loss",
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
        for name in policy_names
    }
    harm = {
        name: float(
            np.mean(
                [row[f"target_{name}_loss"] > row["target_current_loss"] for row in results]
            )
        )
        for name in policy_names
    }
    selected_age = {
        name: _scene_balanced(results, f"{name}_selected_age") for name in policy_names
    }
    common_checks = {
        "exact_coverage": processed_frames
        == int(config["data"]["exact_stream_frames_through_last_query"])
        and len(results) == int(config["data"]["exact_queries"])
        and len({row["scene"] for row in results}) == int(config["data"]["exact_scenes"]),
        "finite": all(math.isfinite(value) for value in means.values()),
        "exact_cached_readout_parity": max(
            row["cached_readout_parity_max_abs"] for row in results
        )
        == 0,
        "exact_capacity": maximum_bank_sizes["reservoir"]
        == int(config["success"]["exact_capacity"])
        and maximum_bank_sizes["fifo"] == int(config["success"]["exact_capacity"]),
    }
    if compute_second_current:
        method_checks = {
            "positive_fifo_agreement_gain_ci95": uncertainty[
                "fifo_agreement_gain_over_current"
            ]["ci95"][0]
            > 0,
            "fifo_better_second_current_ci95": uncertainty[
                "fifo_agreement_over_second_current"
            ]["ci95"][0]
            > 0,
            "fifo_better_reservoir_appearance_ci95": uncertainty[
                "fifo_agreement_over_reservoir_appearance"
            ]["ci95"][0]
            > 0,
            "fifo_better_reservoir_random_ci95": uncertainty[
                "fifo_agreement_over_reservoir_random"
            ]["ci95"][0]
            > 0,
            "fifo_agreement_harm_below_limit": harm["fifo_agreement"]
            <= float(config["success"]["maximum_fifo_agreement_harm_fraction"]),
        }
    else:
        method_checks = {
            "positive_reservoir_agreement_gain_ci95": uncertainty[
                "reservoir_agreement_gain_over_current"
            ]["ci95"][0]
            > 0,
            "reservoir_agreement_better_appearance_ci95": uncertainty[
                "reservoir_agreement_over_appearance"
            ]["ci95"][0]
            > 0,
            "reservoir_agreement_better_random_ci95": uncertainty[
                "reservoir_agreement_over_random"
            ]["ci95"][0]
            > 0,
            "reservoir_agreement_better_fifo_ci95": uncertainty[
                "reservoir_agreement_over_fifo"
            ]["ci95"][0]
            > 0,
            "reservoir_agreement_harm_below_limit": harm["reservoir_agreement"]
            <= float(config["success"]["maximum_reservoir_agreement_harm_fraction"]),
        }
    checks = {**common_checks, **method_checks}
    result = {
        "experiment": config["experiment"],
        "stage": config["purpose"],
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "processed_frames": processed_frames,
        "queries": len(results),
        "scenes": len(scenes),
        "maximum_bank_sizes": maximum_bank_sizes,
        "means": means,
        "uncertainty": uncertainty,
        "acceptance_fractions": acceptance,
        "harm_fractions": harm,
        "scene_balanced_selected_ages": selected_age,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "pose_used": False,
        "pair_identity_used_for_retrieval": False,
        "write_every_frame": True,
        "predict_before_write": True,
        "terminal_accessed": False,
        "rows": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "processed_frames": processed_frames,
                "queries": len(results),
                "means": means,
                "uncertainty": uncertainty,
                "acceptance": acceptance,
                "harm": harm,
                "selected_age": selected_age,
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
