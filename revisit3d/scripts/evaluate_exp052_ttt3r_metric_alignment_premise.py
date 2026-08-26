#!/usr/bin/env python3
"""Zero-fit train-only metric-alignment premise for EXP-052."""
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
from torch.nn import functional as F

from revisit3d.backbones import (
    FrozenCUT3RCarrier,
    patch_center_points,
    symmetric_point_consistency,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _model_view(view: dict, index: int) -> dict:
    """Batch one official 7Scenes view without exposing GT to the model."""
    return {
        "img": view["img"].unsqueeze(0),
        "true_shape": torch.from_numpy(view["true_shape"]).unsqueeze(0),
        "img_mask": torch.tensor([True]),
        "ray_mask": torch.tensor([False]),
        "update": torch.tensor([True]),
        "reset": torch.tensor([index == 0]),
        "idx": index,
        "instance": view["instance"],
    }


def _relative_point_loss(
    predicted: torch.Tensor,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    minimum_depth: float,
    maximum_depth: float,
) -> torch.Tensor:
    if predicted.shape[0] != 1 or predicted.shape[-1] != 3:
        raise ValueError("EXP-052 expects one dense self-view point map")
    device = predicted.device
    dtype = predicted.dtype
    target_depth = torch.as_tensor(depth, device=device, dtype=dtype)
    matrix = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    height, width = target_depth.shape
    if predicted.shape[1:3] != (height, width):
        raise ValueError("EXP-052 prediction/metric grids differ")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    target = torch.stack(
        (
            (xx - matrix[0, 2]) / matrix[0, 0] * target_depth,
            (yy - matrix[1, 2]) / matrix[1, 1] * target_depth,
            target_depth,
        ),
        dim=-1,
    )
    prediction = predicted[0]
    valid = (
        (target_depth >= minimum_depth)
        & (target_depth <= maximum_depth)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(prediction).all(dim=-1)
        & (prediction[..., 2] > 1e-6)
    )
    if int(valid.sum()) < 1024:
        raise RuntimeError("EXP-052 has insufficient registered metric pixels")
    # Per-view scale is an evaluation alignment, not a trainable escape route.
    # Detaching it preserves gradients only through geometric residual shape.
    scale = (
        target_depth[valid].median()
        / prediction[..., 2][valid].median().clamp_min(1e-6)
    ).detach()
    relative_epe = torch.linalg.vector_norm(
        scale * prediction[valid] - target[valid], dim=-1
    ) / target_depth[valid].clamp_min(minimum_depth)
    return relative_epe.mean()


def _rms_normalize(gradient: torch.Tensor) -> torch.Tensor:
    return gradient / gradient.square().mean().sqrt().clamp_min(1e-12)


def _scene_means(rows: list[dict], key: str) -> dict[str, float]:
    return {
        scene: float(np.mean([row[key] for row in rows if row["scene"] == scene]))
        for scene in sorted({row["scene"] for row in rows})
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-052_ttt3r_metric_alignment_premise_v10.yaml"
    )
    parser.add_argument("--confirm-train-rgbd-premise", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgbd_premise or not torch.cuda.is_available():
        raise SystemExit("EXP-052 requires train-RGB-D confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    manifest_path = Path(config["data"]["train_manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-052 result already exists")
    manifest = json.loads(manifest_path.read_text())
    depth_preparation_path = config["output"].get("depth_preparation")
    depth_preparation = (
        json.loads(Path(depth_preparation_path).read_text())
        if depth_preparation_path is not None
        else None
    )
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
        and set(config["data"]["sequences"]).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and (
            depth_preparation is None
            or (
                depth_preparation["selected_frames"]
                == len(config["data"]["sequences"])
                * len(config["data"]["target_frames"])
                * int(config["data"]["context_frames"])
                and depth_preparation["validation_accessed"] is False
                and depth_preparation["terminal_accessed"] is False
            )
        )
    ):
        raise RuntimeError("EXP-052 source-safe contract failed")

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
    carrier.residual.requires_grad_(True)

    rows = []
    root = config["data"]["root"]
    context_frames = int(config["data"]["context_frames"])
    patch_size = int(config["plasticity"]["patch_size"])
    step_size = float(config["plasticity"]["normalized_step"])
    minimum_depth = float(config["metric"]["minimum_depth_m"])
    maximum_depth = float(config["metric"]["maximum_depth_m"])
    torch.cuda.reset_peak_memory_stats()

    for sequence in config["data"]["sequences"]:
        scene = sequence.split("/", 1)[0]
        for target_frame in config["data"]["target_frames"]:
            indices = list(range(int(target_frame) - context_frames + 1, int(target_frame) + 1))
            tuple_spec = sequence + " " + " ".join(f"{index:06d}" for index in indices)
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
            assert previous_prediction is not None and base_prediction is not None and auxiliary is not None
            previous_points = patch_center_points(
                previous_prediction["pts3d_in_other_view"], patch_size
            ).detach()
            tokens = auxiliary["decoder_patch_tokens"].shape[1]
            zero = torch.zeros(
                1, tokens, carrier.code_dim, device="cuda", requires_grad=True
            )
            zero_prediction = carrier.readout(auxiliary, code=zero)
            zero_parity = max(
                float(
                    (
                        zero_prediction[key].detach().float()
                        - base_prediction[key].detach().float()
                    )
                    .abs()
                    .max()
                )
                for key in ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")
                if key in zero_prediction and key in base_prediction
            )
            current_points = patch_center_points(
                zero_prediction["pts3d_in_other_view"], patch_size
            )
            online_loss = symmetric_point_consistency(current_points, previous_points)
            metric_loss = _relative_point_loss(
                zero_prediction["pts3d_in_self_view"],
                gt_views[-1]["depthmap"],
                gt_views[-1]["camera_intrinsics"],
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            online_gradient = torch.autograd.grad(
                online_loss, zero, create_graph=True, retain_graph=True
            )[0]
            metric_gradient = torch.autograd.grad(
                metric_loss, zero, create_graph=False, retain_graph=True
            )[0]
            alignment = float(
                F.cosine_similarity(
                    online_gradient.detach().flatten(),
                    metric_gradient.detach().flatten(),
                    dim=0,
                )
            )

            online_code = zero - step_size * _rms_normalize(online_gradient)
            online_prediction = carrier.readout(auxiliary, code=online_code)
            online_after = symmetric_point_consistency(
                patch_center_points(online_prediction["pts3d_in_other_view"], patch_size),
                previous_points,
            )
            online_metric = _relative_point_loss(
                online_prediction["pts3d_in_self_view"],
                gt_views[-1]["depthmap"],
                gt_views[-1]["camera_intrinsics"],
                minimum_depth=minimum_depth,
                maximum_depth=maximum_depth,
            )
            meta_gradient = torch.autograd.grad(
                online_metric, carrier.residual.projection.weight, create_graph=False
            )[0]

            metric_code = zero.detach() - step_size * _rms_normalize(
                metric_gradient.detach()
            )
            with torch.no_grad():
                oracle_prediction = carrier.readout(auxiliary, code=metric_code)
                oracle_metric = _relative_point_loss(
                    oracle_prediction["pts3d_in_self_view"],
                    gt_views[-1]["depthmap"],
                    gt_views[-1]["camera_intrinsics"],
                    minimum_depth=minimum_depth,
                    maximum_depth=maximum_depth,
                )

            row = {
                "scene": scene,
                "sequence": sequence,
                "target_frame": int(target_frame),
                "zero_code_max_abs_difference": zero_parity,
                "online_loss_before": float(online_loss.detach()),
                "online_loss_after": float(online_after.detach()),
                "online_loss_gain": float((online_loss - online_after).detach()),
                "metric_before": float(metric_loss.detach()),
                "metric_after_online": float(online_metric.detach()),
                "metric_gain_online": float((metric_loss - online_metric).detach()),
                "metric_after_oracle": float(oracle_metric.detach()),
                "metric_gain_oracle": float((metric_loss - oracle_metric).detach()),
                "online_metric_gradient_cosine": alignment,
                "online_gradient_norm": float(online_gradient.detach().norm()),
                "metric_gradient_norm": float(metric_gradient.detach().norm()),
                "exact_meta_gradient_norm": float(meta_gradient.detach().norm()),
                "exact_meta_gradient_finite": bool(torch.isfinite(meta_gradient).all()),
            }
            rows.append(row)
            print(json.dumps({"evaluated": len(rows), "total": 16, **row}), flush=True)
            del dataset, gt_views, model_views, state, previous_prediction, auxiliary
            del base_prediction, zero_prediction, online_prediction, oracle_prediction
            del zero, online_gradient, metric_gradient, meta_gradient, online_code, metric_code
            gc.collect()
            torch.cuda.empty_cache()

    keys = (
        "online_loss_gain",
        "metric_gain_online",
        "metric_gain_oracle",
        "online_metric_gradient_cosine",
        "exact_meta_gradient_norm",
    )
    scene_means = {key: _scene_means(rows, key) for key in keys}
    means = {key: float(np.mean(list(values.values()))) for key, values in scene_means.items()}
    checks = {
        "exact_coverage": len(rows) == int(config["success"]["exact_anchors"])
        and len({row["scene"] for row in rows}) == int(config["success"]["exact_scenes"]),
        "finite": all(
            math.isfinite(value)
            for row in rows
            for key, value in row.items()
            if isinstance(value, float)
        ),
        "zero_code_parity": max(row["zero_code_max_abs_difference"] for row in rows)
        <= float(config["success"]["maximum_zero_code_abs_difference"]),
        "online_loss_descent_all_scenes": all(
            value > 0 for value in scene_means["online_loss_gain"].values()
        ),
        "metric_oracle_gain_all_scenes": all(
            value > 0 for value in scene_means["metric_gain_oracle"].values()
        ),
        "finite_nonzero_exact_meta_gradient_all_anchors": all(
            row["exact_meta_gradient_finite"]
            and row["exact_meta_gradient_norm"] > 0
            for row in rows
        ),
        "no_parameter_update": True,
        "no_validation_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-052",
        "stage": "train_only_zero_fit_metric_alignment_premise",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "means": means,
        "scene_means": scene_means,
        "online_metric_harm_fraction": float(
            np.mean([row["metric_gain_online"] < 0 for row in rows])
        ),
        "gradient_conflict_fraction": float(
            np.mean([row["online_metric_gradient_cosine"] < 0 for row in rows])
        ),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "basis_fit_performed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
