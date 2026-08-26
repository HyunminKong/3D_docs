#!/usr/bin/env python3
"""Train-only zero-fit conditional-tangent capacity audit for EXP-054."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
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
    _model_view,
    _relative_point_loss,
    _rms_normalize,
    _scene_means,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


POLICIES = ("global", "axis_oracle", "token_axis_oracle", "spatial_shuffle")


def _bootstrap_mean(values: list[float], *, samples: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _prediction_difference(first: dict, second: dict) -> float:
    return max(
        float((first[key].detach().float() - second[key].detach().float()).abs().max())
        for key in ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")
        if key in first and key in second
    )


def _policy_result(
    carrier: FrozenCUT3RCarrier,
    auxiliary: dict,
    zero: torch.Tensor,
    online_gradient: torch.Tensor,
    axis_scale: torch.Tensor,
    previous_points: torch.Tensor,
    gt_view: dict,
    *,
    step_size: float,
    patch_size: int,
    minimum_depth: float,
    maximum_depth: float,
    online_before: torch.Tensor,
    metric_before: torch.Tensor,
) -> tuple[dict, dict]:
    scaled_gradient = axis_scale * online_gradient
    code = zero - step_size * _rms_normalize(scaled_gradient)
    with torch.no_grad():
        prediction = carrier.readout(
            auxiliary, code=code.detach(), axis_scale=axis_scale.detach()
        )
        online_after = symmetric_point_consistency(
            patch_center_points(prediction["pts3d_in_other_view"], patch_size),
            previous_points,
        )
        metric_after = _relative_point_loss(
            prediction["pts3d_in_self_view"],
            gt_view["depthmap"],
            gt_view["camera_intrinsics"],
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
        )
    return prediction, {
        "online_loss_after": float(online_after),
        "online_loss_gain": float(online_before.detach() - online_after),
        "metric_after": float(metric_after),
        "metric_gain": float(metric_before.detach() - metric_after),
        "active_fraction": float((axis_scale > 0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-054_conditional_tangent_oracle_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-oracle", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_oracle or not torch.cuda.is_available():
        raise SystemExit("EXP-054 requires train-RGB-D confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    depth_path = Path(config["data"]["depth_preparation"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-054 result already exists")
    manifest = json.loads(manifest_path.read_text())
    depth_preparation = json.loads(depth_path.read_text())
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and depth_preparation["selected_frames"]
        == len(config["data"]["sequences"])
        * len(config["data"]["target_frames"])
        * int(config["data"]["context_frames"])
        and depth_preparation["validation_accessed"] is False
        and depth_preparation["terminal_accessed"] is False
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-054 source-safe contract failed")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    repository = Path(config["carrier"]["repository"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    sys.path.insert(0, str(repository / "src"))
    from eval.mv_recon.data import SevenScenes

    carrier = FrozenCUT3RCarrier(
        checkpoint,
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

    root = config["data"]["root"]
    context_frames = int(config["data"]["context_frames"])
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame in config["data"]["target_frames"]:
            indices = list(
                range(int(target_frame) - context_frames + 1, int(target_frame) + 1)
            )
            tuple_spec = sequence + " " + " ".join(
                f"{index:06d}" for index in indices
            )
            dataset = SevenScenes(
                split="train",
                ROOT=root,
                resolution=tuple(config["carrier"]["resolution"]),
                tuple_list=[tuple_spec],
                seed=seed,
            )
            gt_views = dataset[0]
            model_views = [_model_view(view, index) for index, view in enumerate(gt_views)]

            state = None
            previous_prediction = None
            auxiliary = None
            base_prediction = None
            with torch.no_grad():
                for index, view in enumerate(model_views):
                    prediction, state, current_auxiliary = carrier.step(view, state)
                    if index == len(model_views) - 2:
                        previous_prediction = prediction
                    if index == len(model_views) - 1:
                        base_prediction = prediction
                        auxiliary = current_auxiliary
            assert previous_prediction is not None and base_prediction is not None
            assert auxiliary is not None
            previous_points = patch_center_points(
                previous_prediction["pts3d_in_other_view"], patch_size
            ).detach()
            tokens = auxiliary["decoder_patch_tokens"].float()
            zero = torch.zeros(
                1, tokens.shape[1], carrier.code_dim, device="cuda", requires_grad=True
            )
            scales = conditioner(tokens)
            zero_prediction = carrier.readout(auxiliary, code=zero, axis_scale=scales)
            zero_parity = _prediction_difference(zero_prediction, base_prediction)
            current_points = patch_center_points(
                zero_prediction["pts3d_in_other_view"], patch_size
            )
            online_before = symmetric_point_consistency(current_points, previous_points)
            metric_before = _relative_point_loss(
                zero_prediction["pts3d_in_self_view"],
                gt_views[-1]["depthmap"],
                gt_views[-1]["camera_intrinsics"],
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            online_gradient_graph = torch.autograd.grad(
                online_before, zero, create_graph=True, retain_graph=True
            )[0]
            metric_gradient = torch.autograd.grad(
                metric_before, zero, create_graph=False, retain_graph=True
            )[0].detach()

            conditioned_code = zero - step_size * _rms_normalize(online_gradient_graph)
            conditioned_prediction = carrier.readout(
                auxiliary, code=conditioned_code, axis_scale=scales
            )
            conditioned_metric = _relative_point_loss(
                conditioned_prediction["pts3d_in_self_view"],
                gt_views[-1]["depthmap"],
                gt_views[-1]["camera_intrinsics"],
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            meta_gradient = torch.autograd.grad(
                conditioned_metric, conditioner.projection.weight, create_graph=False
            )[0]
            online_gradient = online_gradient_graph.detach()

            ones = torch.ones_like(online_gradient)
            product = online_gradient * metric_gradient
            axis_mask = (product.sum(dim=1, keepdim=True) > 0).to(online_gradient.dtype)
            axis_mask = axis_mask.expand_as(online_gradient)
            token_axis_mask = (product > 0).to(online_gradient.dtype)
            generator = torch.Generator(device="cpu").manual_seed(seed + len(rows))
            permutation = torch.randperm(
                token_axis_mask.shape[1], generator=generator
            ).to(token_axis_mask.device)
            shuffled_mask = token_axis_mask[:, permutation]
            masks = {
                "global": ones,
                "axis_oracle": axis_mask,
                "token_axis_oracle": token_axis_mask,
                "spatial_shuffle": shuffled_mask,
            }

            policy_predictions = {}
            policy_results = {}
            for policy, mask in masks.items():
                policy_predictions[policy], policy_results[policy] = _policy_result(
                    carrier,
                    auxiliary,
                    zero.detach(),
                    online_gradient,
                    mask,
                    previous_points,
                    gt_views[-1],
                    step_size=step_size,
                    patch_size=patch_size,
                    minimum_depth=minimum_depth,
                    maximum_depth=maximum_depth,
                    online_before=online_before,
                    metric_before=metric_before,
                )

            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": int(target_frame),
                "zero_code_max_abs_difference": zero_parity,
                "conditioner_global_max_abs_difference": _prediction_difference(
                    conditioned_prediction, policy_predictions["global"]
                ),
                "online_loss_before": float(online_before.detach()),
                "metric_before": float(metric_before.detach()),
                "conditioner_meta_gradient_norm": float(meta_gradient.detach().norm()),
                "conditioner_meta_gradient_finite": bool(
                    torch.isfinite(meta_gradient).all()
                ),
                "policies": policy_results,
            }
            rows.append(row)
            print(json.dumps({"evaluated": len(rows), "total": 16, **row}), flush=True)

            del dataset, gt_views, model_views, state, previous_prediction, auxiliary
            del base_prediction, zero_prediction, conditioned_prediction, policy_predictions
            del zero, scales, online_gradient_graph, online_gradient, metric_gradient
            del meta_gradient, conditioned_code, conditioned_metric, masks, product
            gc.collect()
            torch.cuda.empty_cache()

    scene_means = {
        policy: {
            metric: _scene_means(
                [
                    {"scene": row["scene"], metric: row["policies"][policy][metric]}
                    for row in rows
                ],
                metric,
            )
            for metric in ("online_loss_gain", "metric_gain", "active_fraction")
        }
        for policy in POLICIES
    }
    means = {
        policy: {
            metric: float(np.mean(list(scene_means[policy][metric].values())))
            for metric in scene_means[policy]
        }
        for policy in POLICIES
    }
    paired = {}
    for offset, control in enumerate(("global", "spatial_shuffle")):
        values = [
            row["policies"]["token_axis_oracle"]["metric_gain"]
            - row["policies"][control]["metric_gain"]
            for row in rows
        ]
        paired[f"token_axis_minus_{control}"] = _bootstrap_mean(
            values,
            samples=int(config["bootstrap"]["samples"]),
            seed=int(config["bootstrap"]["seed"]) + offset,
        )

    token_harm = float(
        np.mean(
            [row["policies"]["token_axis_oracle"]["metric_gain"] < 0 for row in rows]
        )
    )
    checks = {
        "exact_coverage": len(rows) == int(config["success"]["exact_anchors"])
        and len({row["scene"] for row in rows})
        == int(config["success"]["exact_scenes"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for value in (
                row["zero_code_max_abs_difference"],
                row["conditioner_global_max_abs_difference"],
                row["online_loss_before"],
                row["metric_before"],
                row["conditioner_meta_gradient_norm"],
                *(
                    item
                    for policy in POLICIES
                    for item in row["policies"][policy].values()
                ),
            )
        ),
        "zero_code_parity": max(row["zero_code_max_abs_difference"] for row in rows)
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "zero_conditioner_global_parity": max(
            row["conditioner_global_max_abs_difference"] for row in rows
        )
        <= float(config["success"]["maximum_conditioner_global_abs_difference"]),
        "token_axis_online_descent_all_scenes": all(
            value > 0
            for value in scene_means["token_axis_oracle"]["online_loss_gain"].values()
        ),
        "token_axis_metric_gain_all_scenes": all(
            value > 0
            for value in scene_means["token_axis_oracle"]["metric_gain"].values()
        ),
        "token_axis_beats_global_all_scenes": all(
            scene_means["token_axis_oracle"]["metric_gain"][scene]
            > scene_means["global"]["metric_gain"][scene]
            for scene in scene_means["global"]["metric_gain"]
        ),
        "token_axis_beats_shuffle_all_scenes": all(
            scene_means["token_axis_oracle"]["metric_gain"][scene]
            > scene_means["spatial_shuffle"]["metric_gain"][scene]
            for scene in scene_means["spatial_shuffle"]["metric_gain"]
        ),
        "token_axis_vs_global_positive_ci": paired[
            "token_axis_minus_global"
        ]["ci95"][0]
        > 0,
        "token_axis_vs_shuffle_positive_ci": paired[
            "token_axis_minus_spatial_shuffle"
        ]["ci95"][0]
        > 0,
        "token_axis_harm_within_bound": token_harm
        <= float(config["success"]["maximum_token_axis_harm_fraction"]),
        "token_axis_beats_axis_oracle_mean": means["token_axis_oracle"]["metric_gain"]
        > means["axis_oracle"]["metric_gain"],
        "finite_nonzero_conditioner_meta_gradient_all_anchors": all(
            row["conditioner_meta_gradient_finite"]
            and row["conditioner_meta_gradient_norm"] > 0
            for row in rows
        ),
        "no_parameter_update": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-054",
        "stage": "train_only_zero_fit_conditional_tangent_capacity",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "conditioner_parameters": sum(p.numel() for p in conditioner.parameters()),
        "complete_module_parameters": sum(p.numel() for p in conditioner.parameters())
        + sum(p.numel() for p in carrier.residual.parameters()),
        "means": means,
        "scene_means": scene_means,
        "paired_comparisons": paired,
        "token_axis_harm_fraction": token_harm,
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "conditioner_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
