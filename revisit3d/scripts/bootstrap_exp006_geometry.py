#!/usr/bin/env python3
"""Train and cross-fit the deployable EXP-006 base pose/confidence heads.

Only train-split context frames are accepted.  Frozen VGGT camera/depth outputs
are offline bootstrap targets; the saved custom head is the only runtime
geometry predictor.  The exposed EXP-005 test split is not an argument here.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch import Tensor
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTDepthTeacher, FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import (
    confidence_target,
    fit_confidence_quantiles,
    grouped_folds,
    pose_distillation_loss,
    pose_metrics,
    require_exp006_split,
)
from revisit3d.losses import track_3d_consistency_loss
from revisit3d.models import build_geometry_head
from revisit3d.scripts.train_oracle_revisit import to_device


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("experiment") != "EXP-006":
        raise ValueError("bootstrap requires an EXP-006 configuration")
    require_exp006_split(config["data"]["split"])
    if config["stage0"]["source_head_type"] != "anchored":
        raise ValueError("EXP-006 Stage 0 is pre-registered with the anchored source head")
    return config


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_grid(height: int, width: int, side: int, device: torch.device) -> Tensor:
    ys = (torch.arange(side, device=device) + 0.5) * height / side
    xs = (torch.arange(side, device=device) + 0.5) * width / side
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)


def _release(module: torch.nn.Module) -> None:
    module.cpu()
    del module
    gc.collect()
    torch.cuda.empty_cache()


def _base_cache_pass(dataset: RevisitEpisodeDataset, config: dict, device: torch.device) -> list[dict]:
    foundation = config["foundation"]
    stage0 = config["stage0"]
    checkpoint = torch.load(stage0["source_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("head_type") != "anchored":
        raise RuntimeError("source checkpoint is not an anchored geometry head")
    extractor = FrozenVGGTFeatures(foundation["checkpoint"], repo_root=foundation["repository"]).to(device)
    head = build_geometry_head("anchored", extractor.feature_dim).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    rows: list[dict] = []
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            segment = to_device(sample["a"], str(device))
            features = extractor(segment["context"]["rgb"])
            token = head.token_trunk(head.input_norm(features))
            depth = F.softplus(head.depth_head(token)) + 1e-4
            side = int(math.sqrt(depth.shape[2]))
            if side * side != depth.shape[2]:
                raise RuntimeError("base depth token grid must be square")
            rows.append({
                "episode_id": sample["episode_id"],
                "record_index": index,
                "token": token.cpu().half(),
                "base_depth": depth.squeeze(-1).reshape(1, depth.shape[1], side, side).cpu(),
                "intrinsics": segment["context"]["intrinsics"].cpu(),
            })
            print(json.dumps({"cache": "base", "index": index, "episode": sample["episode_id"]}), flush=True)
    _release(extractor)
    _release(head)
    return rows


def _teacher_cache_pass(dataset: RevisitEpisodeDataset, rows: list[dict], config: dict, device: torch.device) -> None:
    foundation = config["foundation"]
    teacher = FrozenVGGTDepthTeacher(foundation["checkpoint"], repo_root=foundation["repository"]).to(device)
    output_size = (16, 16)
    for index, sample in enumerate(dataset):
        segment = to_device(sample["a"], str(device))
        target = teacher(segment["context"]["rgb"], output_size)
        rows[index].update({
            "teacher_depth": target["depth"].cpu(),
            "teacher_confidence_raw": target["confidence"].cpu(),
            "teacher_w2c": target["w2c"].cpu(),
        })
        print(json.dumps({"cache": "teacher", "index": index, "episode": sample["episode_id"]}), flush=True)
    _release(teacher)


def _tracker_cache_pass(dataset: RevisitEpisodeDataset, rows: list[dict], config: dict, device: torch.device) -> None:
    foundation = config["foundation"]
    side = int(config["stage0"]["track_side"])
    tracker = FrozenVGGTGeometryTracker(foundation["checkpoint"], repo_root=foundation["repository"]).to(device)
    for index, sample in enumerate(dataset):
        segment = to_device(sample["a"], str(device))
        images = segment["context"]["rgb"]
        query = _query_grid(images.shape[-2], images.shape[-1], side, device)
        prior = tracker(images, query)
        # Deliberately discard teacher camera and intrinsics.  Runtime geometry
        # uses the custom pose and calibrated dataset intrinsics.
        rows[index].update({
            "track": prior["track"].cpu().half(),
            "track_visibility": prior["visibility"].cpu().half(),
            "track_confidence": prior["confidence"].cpu().half(),
            "image_size": tuple(images.shape[-2:]),
        })
        print(json.dumps({"cache": "tracker", "index": index, "episode": sample["episode_id"]}), flush=True)
    _release(tracker)


def build_cache(dataset: RevisitEpisodeDataset, config: dict, device: torch.device) -> dict:
    rows = _base_cache_pass(dataset, config, device)
    _teacher_cache_pass(dataset, rows, config, device)
    _tracker_cache_pass(dataset, rows, config, device)
    stage0 = config["stage0"]
    payload = {
        "experiment": "EXP-006",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "manifest": config["data"]["manifest"],
        "source_checkpoint": stage0["source_checkpoint"],
        "source_checkpoint_sha256": _sha256(stage0["source_checkpoint"]),
        "foundation_checkpoint": config["foundation"]["checkpoint"],
        "foundation_checkpoint_sha256": _sha256(config["foundation"]["checkpoint"]),
        "rows": rows,
    }
    cache_path = Path(stage0["cache"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _new_head(config: dict, device: torch.device) -> torch.nn.Module:
    stage0 = config["stage0"]
    checkpoint = torch.load(stage0["source_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("head_type") != "anchored":
        raise RuntimeError("source checkpoint head_type changed after protocol registration")
    head = build_geometry_head("anchored", int(config["foundation"]["feature_dim"])).to(device)
    head.load_state_dict(checkpoint["head"])
    head.requires_grad_(False)
    head.pose_head.requires_grad_(True)
    head.confidence_head.requires_grad_(True)
    return head


def _prediction_from_token(head: torch.nn.Module, token: Tensor) -> tuple[Tensor, Tensor]:
    return head.pose_head(token.mean(dim=2)), torch.sigmoid(head.confidence_head(token)).squeeze(-1)


def _motion_threshold(rows: list[dict]) -> float:
    norms = []
    for row in rows:
        target = row["teacher_w2c"] @ torch.linalg.inv(row["teacher_w2c"][:, :1])
        norms.append(target[:, 1:, :3, 3].norm(dim=-1).flatten())
    median = torch.cat(norms).median()
    return float((0.01 * median).clamp_min(1e-6))


def train_head(
    cache_rows: list[dict], fit_indices: list[int], config: dict, device: torch.device,
    q_low: float, q_high: float, motion_threshold: float, seed: int,
) -> tuple[torch.nn.Module, list[dict]]:
    stage0 = config["stage0"]
    head = _new_head(config, device)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=float(stage0["learning_rate"]), weight_decay=float(stage0["weight_decay"])
    )
    rng = random.Random(seed)
    order: list[int] = []
    logs = []
    conf_cfg = stage0["teacher_confidence"]
    for step in range(int(stage0["steps"])):
        if not order:
            order = list(fit_indices)
            rng.shuffle(order)
        row = cache_rows[order.pop()]
        token = row["token"].to(device=device, dtype=torch.float32)
        target_w2c = row["teacher_w2c"].to(device)
        target_confidence = confidence_target(
            row["teacher_confidence_raw"].to(device), q_low, q_high, float(conf_cfg["epsilon"])
        ).flatten(2)
        predicted_twist, predicted_confidence = _prediction_from_token(head, token)
        pose_loss, terms = pose_distillation_loss(predicted_twist, target_w2c, motion_threshold)
        confidence_loss = F.smooth_l1_loss(predicted_confidence, target_confidence)
        loss = pose_loss + 0.1 * confidence_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(stage0["gradient_clip"]))
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            record = {
                "step": step + 1,
                "episode": row["episode_id"],
                "loss": float(loss.detach()),
                "pose_loss": float(pose_loss.detach()),
                "confidence_loss": float(confidence_loss.detach()),
                "rotation_loss": float(terms["rotation"].detach()),
                "direction_loss": float(terms["translation_direction"].detach()),
                "scale_loss": float(terms["translation_scale"].detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            logs.append(record)
            print(json.dumps({"train": record}), flush=True)
    return head.eval(), logs


def _safe_spearman(prediction: Tensor, target: Tensor) -> float:
    value = spearmanr(prediction.detach().cpu().flatten().numpy(), target.detach().cpu().flatten().numpy()).statistic
    # Undefined correlation means the head or target is constant and must fail
    # the calibration gate.  A finite sentinel keeps the JSON strict.
    return float(value) if math.isfinite(value) else -1.0


def evaluate_head(
    head: torch.nn.Module, cache_rows: list[dict], indices: list[int], group_of: list[str],
    config: dict, device: torch.device, q_low: float, q_high: float, motion_threshold: float, fold: int,
) -> list[dict]:
    conf_cfg = config["stage0"]["teacher_confidence"]
    results = []
    for index in indices:
        row = cache_rows[index]
        token = row["token"].to(device=device, dtype=torch.float32)
        base_depth = row["base_depth"].to(device)
        teacher_w2c = row["teacher_w2c"].to(device)
        target_confidence = confidence_target(
            row["teacher_confidence_raw"].to(device), q_low, q_high, float(conf_cfg["epsilon"])
        ).flatten(2)
        with torch.no_grad():
            predicted_twist, predicted_confidence = _prediction_from_token(head, token)
            metrics = pose_metrics(predicted_twist, teacher_w2c, motion_threshold)
        intrinsics = row["intrinsics"].to(device)
        track = row["track"].to(device=device, dtype=torch.float32)
        visibility = row["track_visibility"].to(device=device, dtype=torch.float32)
        track_confidence = row["track_confidence"].to(device=device, dtype=torch.float32)
        image_size = tuple(row["image_size"])
        predicted_w2c = metrics["predicted_w2c"].detach()
        identity_w2c = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand_as(predicted_w2c)
        predicted_track_loss = track_3d_consistency_loss(
            base_depth, intrinsics, predicted_w2c, track, visibility, track_confidence, image_size=image_size
        )
        teacher_w2c_absolute = row["teacher_w2c"].to(device)
        teacher_w2c = teacher_w2c_absolute @ torch.linalg.inv(teacher_w2c_absolute[:, :1])
        teacher_track_loss = track_3d_consistency_loss(
            base_depth, intrinsics, teacher_w2c, track, visibility, track_confidence, image_size=image_size
        )
        identity_track_loss = track_3d_consistency_loss(
            base_depth, intrinsics, identity_w2c, track, visibility, track_confidence, image_size=image_size
        )
        with torch.enable_grad():
            residual = torch.zeros_like(base_depth, requires_grad=True)
            residual_loss = track_3d_consistency_loss(
                base_depth.detach() * residual.exp(), intrinsics, predicted_w2c, track,
                visibility, track_confidence, image_size=image_size,
            )
            residual_gradient, = torch.autograd.grad(residual_loss, residual)
        rotation = metrics["rotation_error_deg"]
        direction = metrics["translation_direction_error_deg"]
        finite = all(torch.isfinite(value).all().item() for value in (
            base_depth, predicted_twist, predicted_confidence, predicted_w2c,
            predicted_track_loss, teacher_track_loss, identity_track_loss, residual_gradient,
        ))
        result = {
            "episode": row["episode_id"],
            "component": group_of[index],
            "fold": fold,
            "finite": bool(finite),
            "positive_depth_fraction": float((base_depth > 0).float().mean()),
            "view0_identity_error": float(metrics["view0_identity_error"]),
            "median_rotation_error_deg": float(rotation.median()),
            "median_translation_direction_error_deg": (
                float(direction.median()) if direction.numel() else 180.0
            ),
            "scale_aligned_translation_error": (
                float(metrics["scale_aligned_translation_error"])
                if torch.isfinite(metrics["scale_aligned_translation_error"]) else 1e6
            ),
            "confidence_spearman": _safe_spearman(predicted_confidence, target_confidence),
            "predicted_track_loss": float(predicted_track_loss),
            "teacher_track_loss": float(teacher_track_loss),
            "identity_track_loss": float(identity_track_loss),
            "depth_residual_gradient_norm": float(residual_gradient.norm()),
            "depth_residual_gradient_healthy": bool(
                torch.isfinite(residual_gradient).all() and residual_gradient.norm() > 1e-12
            ),
        }
        results.append(result)
        print(json.dumps({"crossfit": result}), flush=True)
    return results


def _component_mean(rows: list[dict], key: str) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["component"], []).append(float(row[key]))
    values = [float(np.mean(group)) for group in grouped.values()]
    return float(np.mean(values))


def summarize_health(rows: list[dict], config: dict) -> tuple[dict, dict]:
    gates = config["stage0"]["gates"]
    summary = {
        "episodes": len(rows),
        "components": len({row["component"] for row in rows}),
        "finite_all": all(row["finite"] for row in rows),
        "positive_depth_fraction": float(np.mean([row["positive_depth_fraction"] for row in rows])),
        "maximum_view0_identity_error": max(row["view0_identity_error"] for row in rows),
        "median_rotation_error_deg": float(np.nanmedian([row["median_rotation_error_deg"] for row in rows])),
        "median_translation_direction_error_deg": float(np.nanmedian([
            row["median_translation_direction_error_deg"] for row in rows
        ])),
        "median_scale_aligned_translation_error": float(np.nanmedian([
            row["scale_aligned_translation_error"] for row in rows
        ])),
        "component_mean_confidence_spearman": _component_mean(rows, "confidence_spearman"),
        "component_mean_predicted_track_loss": _component_mean(rows, "predicted_track_loss"),
        "component_mean_teacher_track_loss": _component_mean(rows, "teacher_track_loss"),
        "component_mean_identity_track_loss": _component_mean(rows, "identity_track_loss"),
        "healthy_depth_gradient_fraction": float(np.mean([
            row["depth_residual_gradient_healthy"] for row in rows
        ])),
    }
    summary["predicted_to_identity_track_loss_ratio"] = (
        summary["component_mean_predicted_track_loss"]
        / max(summary["component_mean_identity_track_loss"], 1e-8)
    )
    summary["predicted_to_teacher_track_loss_ratio"] = (
        summary["component_mean_predicted_track_loss"]
        / max(summary["component_mean_teacher_track_loss"], 1e-8)
    )
    summary["teacher_to_identity_track_loss_ratio"] = (
        summary["component_mean_teacher_track_loss"]
        / max(summary["component_mean_identity_track_loss"], 1e-8)
    )
    checks = {
        "finite": (not bool(gates["finite_required"])) or summary["finite_all"],
        "positive_depth": summary["positive_depth_fraction"] >= gates["minimum_positive_depth_fraction"],
        "view0_identity": summary["maximum_view0_identity_error"] <= gates["maximum_view0_identity_error"],
        "rotation": summary["median_rotation_error_deg"] <= gates["maximum_median_rotation_error_deg"],
        "translation_direction": summary["median_translation_direction_error_deg"] <= gates[
            "maximum_median_translation_direction_error_deg"
        ],
        "translation_scale": summary["median_scale_aligned_translation_error"] <= gates[
            "maximum_median_scale_aligned_translation_error"
        ],
        "confidence": summary["component_mean_confidence_spearman"] >= gates["minimum_confidence_spearman"],
        "track_objective_retention": summary["predicted_to_teacher_track_loss_ratio"] <= gates[
            "maximum_predicted_to_teacher_track_loss_ratio"
        ],
        "depth_gradient": summary["healthy_depth_gradient_fraction"] >= gates[
            "minimum_healthy_depth_gradient_fraction"
        ],
    }
    checks["passed"] = all(checks.values())
    return summary, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 Stage 0 requires CUDA")
    torch.set_float32_matmul_precision("high")
    config = _load_config(args.config)
    device = torch.device("cuda")
    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    cache_path = Path(config["stage0"]["cache"])
    if args.rebuild_cache or not cache_path.exists():
        cache = build_cache(dataset, config, device)
    else:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("split") != "train" or cache.get("experiment") != "EXP-006":
            raise RuntimeError("refusing an incompatible Stage-0 cache")
    if args.cache_only:
        print(json.dumps({"cache": str(cache_path), "rows": len(cache["rows"])}))
        return

    rows = cache["rows"]
    stage0 = config["stage0"]
    conf_cfg = stage0["teacher_confidence"]
    q_low, q_high = fit_confidence_quantiles(
        (row["teacher_confidence_raw"] for row in rows),
        float(conf_cfg["lower_quantile"]), float(conf_cfg["upper_quantile"]), float(conf_cfg["epsilon"]),
    )
    motion_threshold = _motion_threshold(rows)
    fold_indices, group_of = grouped_folds(dataset.records, int(stage0["folds"]), int(config["seed"]))
    generated_fold_groups = [sorted({group_of[index] for index in held_out}) for held_out in fold_indices]
    registered_fold_groups = [sorted(groups) for groups in stage0["fold_components"]]
    if generated_fold_groups != registered_fold_groups:
        raise RuntimeError(
            f"manifest overlap components changed: generated={generated_fold_groups}, "
            f"registered={registered_fold_groups}"
        )
    crossfit_rows = []
    train_logs = []
    all_indices = set(range(len(rows)))
    for fold, held_out in enumerate(fold_indices):
        fit = sorted(all_indices.difference(held_out))
        head, logs = train_head(
            rows, fit, config, device, q_low, q_high, motion_threshold, int(config["seed"]) + fold,
        )
        train_logs.append({"fold": fold, "held_out": held_out, "logs": logs})
        crossfit_rows.extend(evaluate_head(
            head, rows, held_out, group_of, config, device, q_low, q_high, motion_threshold, fold,
        ))
        _release(head)
    summary, gate = summarize_health(crossfit_rows, config)
    print(json.dumps({"stage0_summary": summary, "health_gate": gate}), flush=True)

    final_head, final_logs = train_head(
        rows, list(range(len(rows))), config, device, q_low, q_high, motion_threshold, int(config["seed"]) + 100,
    )
    checkpoint_path = Path(stage0["output_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head": {key: value.detach().cpu() for key, value in final_head.state_dict().items()},
        "head_type": "anchored",
        "experiment": "EXP-006",
        "protocol_revision": config["protocol_revision"],
        "source_checkpoint": stage0["source_checkpoint"],
        "source_checkpoint_sha256": cache["source_checkpoint_sha256"],
        "teacher_confidence_quantiles": {"low": q_low, "high": q_high},
        "motion_threshold": motion_threshold,
        "health_gate": gate,
        "health_summary": summary,
        "runtime_teacher": False,
    }, checkpoint_path)
    _release(final_head)

    fold_groups = generated_fold_groups
    result = {
        "experiment": "EXP-006",
        "stage": 0,
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "config": args.config,
        "cache": str(cache_path),
        "output_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": cache["source_checkpoint_sha256"],
        "foundation_checkpoint_sha256": cache["foundation_checkpoint_sha256"],
        "confidence_quantiles": {"low": q_low, "high": q_high},
        "motion_threshold": motion_threshold,
        "fold_groups": fold_groups,
        "crossfit_rows": crossfit_rows,
        "crossfit_train_logs": train_logs,
        "final_train_logs": final_logs,
        "summary": summary,
        "health_gate": gate,
    }
    result_path = Path(stage0["result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"result": str(result_path), "checkpoint": str(checkpoint_path), "passed": gate["passed"]}))


if __name__ == "__main__":
    main()
