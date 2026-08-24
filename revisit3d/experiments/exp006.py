"""Leakage-safe utilities shared by EXP-006 training and evaluation.

This module contains protocol, target, and metric code only.  In particular it
does not provide an oracle pose fallback for the deployable model path.
"""

from __future__ import annotations

import math
import random
import hashlib
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from revisit3d.losses import relative_w2c_from_twist


def require_exp006_split(split: str, *, allow_validation: bool = False) -> None:
    """Reject the exposed test split at the common EXP-006 boundary."""
    allowed = {"train", "val"} if allow_validation else {"train"}
    if split not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"EXP-006 split must be one of {{{choices}}}; got {split!r}")


def _component_ids(records: Sequence[dict]) -> dict[str, str]:
    scenes = {scene for record in records for scene in (record["source_scene"], record["target_scene"])}
    parent = {scene: scene for scene in scenes}

    def find(scene: str) -> str:
        while parent[scene] != scene:
            parent[scene] = parent[parent[scene]]
            scene = parent[scene]
        return scene

    for record in records:
        left, right = find(record["source_scene"]), find(record["target_scene"])
        if left != right:
            parent[left] = right
    components: dict[str, list[str]] = {}
    for scene in sorted(scenes):
        components.setdefault(find(scene), []).append(scene)
    result = {}
    for members in components.values():
        identifier = "__".join(sorted(members))
        result.update({scene: identifier for scene in members})
    return result


def grouped_folds(records: Sequence[dict], folds: int, seed: int) -> tuple[list[list[int]], list[str]]:
    """Make deterministic folds grouped by physical-overlap component."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    scene_to_component = _component_ids(records)
    group_of = [scene_to_component[record["source_scene"]] for record in records]
    groups = sorted(set(group_of))
    if folds > len(groups):
        raise ValueError(f"{folds} folds requested for only {len(groups)} overlap components")
    random.Random(seed).shuffle(groups)
    fold_groups = [groups[index::folds] for index in range(folds)]
    fold_indices = [[index for index, group in enumerate(group_of) if group in held_out]
                    for held_out in fold_groups]
    if sorted(index for fold in fold_indices for index in fold) != list(range(len(records))):
        raise RuntimeError("grouped folds do not partition the records")
    return fold_indices, group_of


def deterministic_foreign_indices(
    records: Sequence[dict], current_index: int, count: int, seed: int,
) -> list[int]:
    current = records[current_index]
    current_scenes = {current["source_scene"], current["target_scene"]}
    eligible = [
        index for index, record in enumerate(records)
        if index != current_index
        and not current_scenes.intersection({record["source_scene"], record["target_scene"]})
    ]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} foreign episodes are eligible; need {count}")
    eligible.sort(key=lambda index: records[index].get("episode_id", str(index)))
    token = f"{seed}:{current.get('episode_id', current_index)}".encode()
    offset = int(hashlib.sha256(token).hexdigest()[:8], 16) % len(eligible)
    rotated = eligible[offset:] + eligible[:offset]
    return rotated[:count]


def _confidence_logit(raw_confidence: Tensor, epsilon: float) -> Tensor:
    # FastVGGT's DPT confidence activation is expp1: confidence = 1 + exp(logit).
    return torch.log((raw_confidence.float() - 1).clamp_min(epsilon))


def fit_confidence_quantiles(
    values: Iterable[Tensor], lower: float = 0.05, upper: float = 0.95, epsilon: float = 1e-6,
) -> tuple[float, float]:
    if not 0 <= lower < upper <= 1:
        raise ValueError("confidence quantiles must satisfy 0 <= lower < upper <= 1")
    flattened = torch.cat([_confidence_logit(value, epsilon).flatten().cpu() for value in values])
    q_low, q_high = torch.quantile(flattened, torch.tensor([lower, upper])).tolist()
    if not math.isfinite(q_low) or not math.isfinite(q_high) or q_high - q_low <= 1e-6:
        raise RuntimeError(f"degenerate train confidence quantiles: {q_low}, {q_high}")
    return float(q_low), float(q_high)


def confidence_target(raw_confidence: Tensor, q_low: float, q_high: float, epsilon: float = 1e-6) -> Tensor:
    if q_high <= q_low:
        raise ValueError("q_high must exceed q_low")
    target = (_confidence_logit(raw_confidence, epsilon) - q_low) / (q_high - q_low)
    return target.clamp(0, 1)


def relative_w2c(w2c: Tensor) -> Tensor:
    if w2c.ndim != 4 or w2c.shape[-2:] != (4, 4):
        raise ValueError("w2c must be [batch, views, 4, 4]")
    return w2c @ torch.linalg.inv(w2c[:, :1])


def _rotation_geodesic(prediction: Tensor, target: Tensor) -> Tensor:
    relative = prediction @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    vee = torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1],
    ), dim=-1)
    sine = 0.5 * vee.norm(dim=-1)
    return torch.atan2(sine, cosine)


def pose_distillation_loss(
    predicted_twist: Tensor, teacher_w2c: Tensor, motion_threshold: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    predicted = relative_w2c_from_twist(predicted_twist)
    target = relative_w2c(teacher_w2c)
    rotation = _rotation_geodesic(predicted[:, 1:, :3, :3], target[:, 1:, :3, :3]).mean()
    pred_t, target_t = predicted[:, 1:, :3, 3], target[:, 1:, :3, 3]
    pred_norm, target_norm = pred_t.norm(dim=-1), target_t.norm(dim=-1)
    valid = target_norm > motion_threshold
    if valid.any():
        cosine = F.cosine_similarity(pred_t[valid], target_t[valid], dim=-1).clamp(-1, 1)
        direction = (1 - cosine).mean()
        scale = F.smooth_l1_loss(
            torch.log(pred_norm[valid] + 1e-6), torch.log(target_norm[valid] + 1e-6)
        )
    else:
        direction = rotation.new_zeros(())
        scale = rotation.new_zeros(())
    total = rotation + 0.5 * direction + 0.1 * scale
    return total, {"rotation": rotation, "translation_direction": direction, "translation_scale": scale}


def pose_metrics(predicted_twist: Tensor, teacher_w2c: Tensor, motion_threshold: float) -> dict[str, Tensor]:
    predicted = relative_w2c_from_twist(predicted_twist)
    target = relative_w2c(teacher_w2c)
    rotation_deg = torch.rad2deg(
        _rotation_geodesic(predicted[:, 1:, :3, :3], target[:, 1:, :3, :3])
    )
    pred_t, target_t = predicted[:, 1:, :3, 3], target[:, 1:, :3, 3]
    target_norm = target_t.norm(dim=-1)
    valid = target_norm > motion_threshold
    if valid.any():
        cosine = F.cosine_similarity(pred_t[valid], target_t[valid], dim=-1).clamp(-1, 1)
        direction_deg = torch.rad2deg(torch.acos(cosine))
        numerator = (pred_t[valid] * target_t[valid]).sum()
        denominator = pred_t[valid].square().sum().clamp_min(1e-8)
        alpha = numerator / denominator
        aligned = alpha * pred_t[valid]
        scale_aligned = (aligned - target_t[valid]).norm() / target_t[valid].norm().clamp_min(1e-8)
    else:
        direction_deg = rotation_deg.new_empty(0)
        scale_aligned = rotation_deg.new_tensor(float("nan"))
    identity = torch.eye(4, device=predicted.device, dtype=predicted.dtype)
    view0_error = (predicted[:, 0] - identity).abs().max()
    return {
        "rotation_error_deg": rotation_deg,
        "translation_direction_error_deg": direction_deg,
        "scale_aligned_translation_error": scale_aligned,
        "view0_identity_error": view0_error,
        "predicted_w2c": predicted,
    }
