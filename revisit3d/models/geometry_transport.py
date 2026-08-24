"""Predicted-geometry alignment and spatial atom transport for EXP-006."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .plasticity_atom import PlasticityAtom


@dataclass
class FeatureMatches:
    source_index: Tensor
    target_index: Tensor
    similarity: Tensor


@dataclass
class Sim3Alignment:
    scale: Tensor
    rotation: Tensor
    translation: Tensor
    valid: bool
    correspondences: int
    inliers: int
    inlier_ratio: float
    normalized_median_residual: float
    source_rank_ratio: float
    target_rank_ratio: float


@dataclass
class TransportResult:
    code: Tensor
    valid: Tensor
    normalized_entropy: Tensor
    mean_max_weight: Tensor
    coverage: Tensor


def backproject_tokens(
    depth: Tensor,
    intrinsics: Tensor,
    w2c: Tensor,
    *,
    image_size: tuple[int, int],
) -> Tensor:
    """Backproject token-centre depths into the view-0 reference frame."""
    if depth.ndim != 4:
        raise ValueError("depth must be [B,V,H,W] or [B,V,P,1]")
    if depth.shape[-1] == 1:
        points = depth.shape[2]
        side = int(math.sqrt(points))
        if side * side != points:
            raise ValueError("flattened depth must have a square token count")
        grid_h = grid_w = side
        z = depth.squeeze(-1)
    else:
        grid_h, grid_w = depth.shape[-2:]
        z = depth.flatten(2)
    if intrinsics.shape != (*depth.shape[:2], 4) or w2c.shape != (*depth.shape[:2], 4, 4):
        raise ValueError("intrinsics/w2c do not match depth batch and views")
    image_h, image_w = image_size
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, device=depth.device, dtype=depth.dtype),
        torch.arange(grid_w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    u = (xx.flatten() + 0.5) * image_w / grid_w
    v = (yy.flatten() + 0.5) * image_h / grid_h
    fx, fy, cx, cy = intrinsics.to(depth.dtype).unbind(-1)
    x = (u - cx[..., None]) / fx[..., None] * z
    y = (v - cy[..., None]) / fy[..., None] * z
    camera = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
    reference = torch.einsum("bvij,bvnj->bvni", torch.linalg.inv(w2c.to(depth.dtype)), camera)
    return reference[..., :3]


def local_knn_scale(xyz: Tensor, k: int = 8) -> Tensor:
    """Return within-view local surface spacing for every token.

    Views overlap in the common segment reference frame, so pooling views before
    k-NN lets cross-view duplicate observations collapse the bandwidth toward
    zero.  The spacing is a property of each view's sampled surface and is
    therefore estimated independently for every ``[B,V]`` point set.
    """
    if xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError("xyz must be [B,V,P,3]")
    points = xyz.shape[2]
    if points <= k:
        raise ValueError("point set is too small for requested k")
    distance = torch.cdist(xyz.float(), xyz.float())
    diagonal = torch.eye(points, device=distance.device, dtype=torch.bool)[None, None]
    distance = distance.masked_fill(diagonal, float("inf"))
    neighbors = distance.topk(k, dim=-1, largest=False).values
    scale = neighbors.median(dim=-1).values.clamp_min(1e-6)
    return scale.unsqueeze(-1).to(xyz.dtype)


def mutual_feature_matches(
    source_key: Tensor, target_key: Tensor, *, minimum_similarity: float = 0.60,
) -> list[FeatureMatches]:
    if source_key.ndim != 4 or target_key.ndim != 4 or source_key.shape[0] != target_key.shape[0]:
        raise ValueError("source/target keys must be [B,V,P,D] with matching batch")
    result = []
    for source, target in zip(source_key.flatten(1, 2), target_key.flatten(1, 2)):
        similarity = F.normalize(target, dim=-1) @ F.normalize(source, dim=-1).transpose(0, 1)
        source_for_target = similarity.argmax(dim=1)
        target_for_source = similarity.argmax(dim=0)
        target_index = torch.arange(target.shape[0], device=target.device)
        score = similarity[target_index, source_for_target]
        mutual = target_for_source[source_for_target] == target_index
        keep = mutual & (score >= minimum_similarity)
        result.append(FeatureMatches(
            source_index=source_for_target[keep].detach(),
            target_index=target_index[keep].detach(),
            similarity=score[keep].detach(),
        ))
    return result


def weighted_sim3(source: Tensor, target: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted Umeyama transform mapping source points into target points."""
    if source.ndim != 2 or source.shape != target.shape or source.shape[-1] != 3:
        raise ValueError("source and target must be matching [N,3] tensors")
    if weight.shape != source.shape[:1]:
        raise ValueError("weight must be [N]")
    weight = weight.float().clamp_min(0)
    weight = weight / weight.sum().clamp_min(1e-8)
    source_f, target_f = source.float(), target.float()
    source_mean = (weight[:, None] * source_f).sum(dim=0)
    target_mean = (weight[:, None] * target_f).sum(dim=0)
    source_centered = source_f - source_mean
    target_centered = target_f - target_mean
    covariance = source_centered.transpose(0, 1) @ (weight[:, None] * target_centered)
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones(3, device=source.device, dtype=source_f.dtype)
    if torch.det(vh.transpose(0, 1) @ u.transpose(0, 1)) < 0:
        correction[-1] = -1
    rotation = vh.transpose(0, 1) @ torch.diag(correction) @ u.transpose(0, 1)
    variance = (weight * source_centered.square().sum(dim=-1)).sum().clamp_min(1e-8)
    scale = (singular * correction).sum() / variance
    translation = target_mean - scale * (rotation @ source_mean)
    return scale.to(source.dtype), rotation.to(source.dtype), translation.to(source.dtype)


def apply_sim3(points: Tensor, alignment: Sim3Alignment) -> Tensor:
    return alignment.scale * torch.einsum("ij,...j->...i", alignment.rotation, points) + alignment.translation


def _rank_ratio(points: Tensor, weight: Tensor) -> float:
    normalized = weight / weight.sum().clamp_min(1e-8)
    mean = (normalized[:, None] * points).sum(dim=0)
    centered = points - mean
    covariance = centered.transpose(0, 1) @ (normalized[:, None] * centered)
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).flip(0).clamp_min(0)
    return float(eigenvalues[1] / eigenvalues[0].clamp_min(1e-8))


def _invalid_alignment(device: torch.device, dtype: torch.dtype, correspondences: int = 0) -> Sim3Alignment:
    return Sim3Alignment(
        scale=torch.ones((), device=device, dtype=dtype),
        rotation=torch.eye(3, device=device, dtype=dtype),
        translation=torch.zeros(3, device=device, dtype=dtype),
        valid=False,
        correspondences=correspondences,
        inliers=0,
        inlier_ratio=0.0,
        normalized_median_residual=float("inf"),
        source_rank_ratio=0.0,
        target_rank_ratio=0.0,
    )


def robust_sim3(
    source_xyz: Tensor,
    target_xyz: Tensor,
    source_confidence: Tensor,
    target_confidence: Tensor,
    target_scale: Tensor,
    matches: FeatureMatches,
    *,
    minimum_correspondences: int = 32,
    residual_threshold: float = 2.5,
    minimum_inlier_ratio: float = 0.25,
    minimum_rank_ratio: float = 0.01,
    scale_bounds: tuple[float, float] = (0.1, 10.0),
) -> Sim3Alignment:
    count = int(matches.source_index.numel())
    if count < minimum_correspondences:
        return _invalid_alignment(source_xyz.device, source_xyz.dtype, count)
    source = source_xyz[matches.source_index]
    target = target_xyz[matches.target_index]
    confidence = torch.sqrt(
        source_confidence[matches.source_index].flatten().clamp(1e-4, 1)
        * target_confidence[matches.target_index].flatten().clamp(1e-4, 1)
    )
    weight = ((matches.similarity - 0.60) / 0.40).clamp(0, 1) * confidence
    if weight.sum() <= 1e-8:
        return _invalid_alignment(source_xyz.device, source_xyz.dtype, count)
    try:
        scale, rotation, translation = weighted_sim3(source, target, weight)
    except torch.linalg.LinAlgError:
        return _invalid_alignment(source_xyz.device, source_xyz.dtype, count)
    aligned = scale * torch.einsum("ij,nj->ni", rotation, source) + translation
    local = target_scale[matches.target_index].flatten().clamp_min(1e-6)
    normalized_residual = (target - aligned).norm(dim=-1) / local
    inlier = normalized_residual <= residual_threshold
    inliers = int(inlier.sum())
    if inliers < minimum_correspondences:
        return _invalid_alignment(source_xyz.device, source_xyz.dtype, count)
    source, target, weight, local = source[inlier], target[inlier], weight[inlier], local[inlier]
    try:
        scale, rotation, translation = weighted_sim3(source, target, weight)
    except torch.linalg.LinAlgError:
        return _invalid_alignment(source_xyz.device, source_xyz.dtype, count)
    aligned = scale * torch.einsum("ij,nj->ni", rotation, source) + translation
    normalized_residual = (target - aligned).norm(dim=-1) / local
    median_residual = float(normalized_residual.median())
    source_rank = _rank_ratio(source.float(), weight.float())
    target_rank = _rank_ratio(target.float(), weight.float())
    inlier_ratio = inliers / count
    finite = all(torch.isfinite(value).all().item() for value in (scale, rotation, translation))
    valid = (
        finite
        and inlier_ratio >= minimum_inlier_ratio
        and source_rank >= minimum_rank_ratio
        and target_rank >= minimum_rank_ratio
        and scale_bounds[0] <= float(scale) <= scale_bounds[1]
        and float(torch.det(rotation)) > 0
        and median_residual <= residual_threshold
    )
    return Sim3Alignment(
        scale=scale.detach(),
        rotation=rotation.detach(),
        translation=translation.detach(),
        valid=bool(valid),
        correspondences=count,
        inliers=inliers,
        inlier_ratio=float(inlier_ratio),
        normalized_median_residual=median_residual,
        source_rank_ratio=source_rank,
        target_rank_ratio=target_rank,
    )


def align_atoms(source: PlasticityAtom, target: PlasticityAtom) -> list[Sim3Alignment]:
    matches = mutual_feature_matches(source.key, target.key)
    result = []
    for batch, batch_matches in enumerate(matches):
        result.append(robust_sim3(
            source.xyz[batch].flatten(0, 1),
            target.xyz[batch].flatten(0, 1),
            source.confidence[batch].flatten(0, 1),
            target.confidence[batch].flatten(0, 1),
            target.scale[batch].flatten(0, 1),
            batch_matches,
        ))
    return result


def visual_transport(
    source: PlasticityAtom, target: PlasticityAtom, *, temperature: float = 0.07,
) -> TransportResult:
    outputs, entropy, max_weight, coverage = [], [], [], []
    for source_key, target_key, source_code in zip(
        source.key.flatten(1, 2), target.key.flatten(1, 2), source.code.flatten(1, 2),
    ):
        cosine = F.normalize(target_key, dim=-1) @ F.normalize(source_key, dim=-1).transpose(0, 1)
        weight = torch.softmax(cosine / temperature, dim=-1)
        outputs.append(weight @ source_code)
        entropy.append((-(weight * weight.clamp_min(1e-8).log()).sum(-1) / math.log(weight.shape[-1])).mean())
        max_weight.append(weight.max(dim=-1).values.mean())
        coverage.append((cosine.max(dim=-1).values >= 0.60).float().mean())
    shape = target.code.shape
    return TransportResult(
        code=torch.stack(outputs).reshape(shape),
        valid=torch.ones(shape[0], dtype=torch.bool, device=target.code.device),
        normalized_entropy=torch.stack(entropy),
        mean_max_weight=torch.stack(max_weight),
        coverage=torch.stack(coverage),
    )


def geometry_transport(
    source: PlasticityAtom,
    target: PlasticityAtom,
    alignments: list[Sim3Alignment],
    *,
    appearance_weight: float = 0.0,
    neighbors: int = 8,
) -> TransportResult:
    if len(alignments) != source.code.shape[0] or target.code.shape[0] != source.code.shape[0]:
        raise ValueError("one alignment per batch item is required")
    outputs, valid, entropy, max_weight, coverage = [], [], [], [], []
    for batch, alignment in enumerate(alignments):
        target_code = target.code[batch].flatten(0, 1)
        if not alignment.valid:
            outputs.append(torch.zeros_like(target_code))
            valid.append(False)
            entropy.append(target_code.new_tensor(0.0))
            max_weight.append(target_code.new_tensor(0.0))
            coverage.append(target_code.new_tensor(0.0))
            continue
        source_xyz = apply_sim3(source.xyz[batch].flatten(0, 1), alignment)
        target_xyz = target.xyz[batch].flatten(0, 1)
        distance = torch.cdist(target_xyz.float(), source_xyz.float())
        nearest_distance, nearest = distance.topk(neighbors, dim=-1, largest=False)
        nearest = nearest.detach()
        source_scale = source.scale[batch].flatten(0, 1).flatten()[nearest] * alignment.scale.abs()
        target_scale = target.scale[batch].flatten(0, 1).flatten()[:, None]
        denominator = 2 * (target_scale.square() + source_scale.square() + 1e-8)
        logits = -nearest_distance.square() / denominator
        if appearance_weight:
            source_key = source.key[batch].flatten(0, 1)[nearest]
            target_key = target.key[batch].flatten(0, 1)[:, None, :]
            logits = logits + appearance_weight * F.cosine_similarity(target_key, source_key, dim=-1)
        weight = torch.softmax(logits, dim=-1)
        source_code = source.code[batch].flatten(0, 1)[nearest]
        outputs.append((weight[..., None] * source_code).sum(dim=1))
        if neighbors == 1:
            entropy.append(weight.new_zeros(()))
        else:
            entropy.append((-(weight * weight.clamp_min(1e-8).log()).sum(-1) / math.log(neighbors)).mean())
        max_weight.append(weight.max(dim=-1).values.mean())
        normalized_distance = nearest_distance / (target_scale.square() + source_scale.square()).sqrt().clamp_min(1e-6)
        coverage.append((normalized_distance[:, 0] <= 2.5).float().mean())
        valid.append(True)
    shape = target.code.shape
    return TransportResult(
        code=torch.stack(outputs).reshape(shape),
        valid=torch.tensor(valid, dtype=torch.bool, device=target.code.device),
        normalized_entropy=torch.stack(entropy),
        mean_max_weight=torch.stack(max_weight),
        coverage=torch.stack(coverage),
    )
