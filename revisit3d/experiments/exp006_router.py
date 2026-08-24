"""Observable feature contract for the locked EXP-006 utility router."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


DESCRIPTOR_DIMENSIONS = 256
OBSERVABLE_SCALAR_DIMENSIONS = 24
ALIGNMENT_SCALAR_INDICES = (12, 13, 14, 15)
PRIMARY_SCALAR_INDICES = tuple(range(12)) + tuple(range(16, 24))


def observable_router_features(
    *,
    current_descriptor: Tensor,
    source_descriptor: Tensor,
    current_code: Tensor,
    transported_code: Tensor,
    visual_result: object,
    alignment: object,
    current_pre_objective: Tensor,
    current_post_objective: Tensor,
    candidate_objective: Tensor,
    source_pre_objective: Tensor,
    source_post_objective: Tensor,
    current_pre_stats: dict[str, Tensor],
    current_post_stats: dict[str, Tensor],
    source_pre_stats: dict[str, Tensor],
    source_post_stats: dict[str, Tensor],
) -> Tensor:
    """Build the 256+24 observable vector without future/query quantities."""
    descriptor = torch.cat((
        current_descriptor,
        source_descriptor,
        current_descriptor - source_descriptor,
        current_descriptor * source_descriptor,
    ))
    denominator = current_pre_objective.detach().abs().clamp_min(1e-6)
    source_denominator = source_pre_objective.detach().abs().clamp_min(1e-6)
    scalars = torch.stack((
        current_pre_objective / denominator,
        current_post_objective / denominator,
        candidate_objective / denominator,
        (current_post_objective - candidate_objective) / denominator,
        F.cosine_similarity(current_code.flatten(1), transported_code.flatten(1), dim=-1)[0],
        transported_code.abs().mean(),
        transported_code.square().mean().sqrt(),
        transported_code.flatten(1, 2).std(dim=1).mean(),
        visual_result.normalized_entropy[0],
        visual_result.mean_max_weight[0],
        visual_result.coverage[0],
        current_descriptor @ source_descriptor,
        current_descriptor.new_tensor(float(alignment.valid)),
        current_descriptor.new_tensor(alignment.inlier_ratio),
        current_descriptor.new_tensor(
            alignment.normalized_median_residual if alignment.valid else 10.0
        ),
        current_descriptor.new_tensor(alignment.correspondences / 2048.0),
        source_post_objective / source_denominator,
        (source_pre_objective - source_post_objective) / source_denominator,
        source_pre_stats["track_coverage"],
        source_pre_stats["mean_3d_residual"],
        source_post_stats["mean_3d_residual"],
        current_pre_stats["track_coverage"],
        current_pre_stats["mean_3d_residual"],
        current_post_stats["mean_3d_residual"],
    ))
    features = torch.cat((descriptor, scalars))
    expected = DESCRIPTOR_DIMENSIONS + OBSERVABLE_SCALAR_DIMENSIONS
    if features.numel() != expected:
        raise RuntimeError(f"router feature contract produced {features.numel()} values, expected {expected}")
    return features


def primary_feature_columns(
    descriptor_dimensions: int = DESCRIPTOR_DIMENSIONS,
    scalar_indices: tuple[int, ...] = PRIMARY_SCALAR_INDICES,
) -> list[int]:
    if descriptor_dimensions != DESCRIPTOR_DIMENSIONS:
        raise ValueError("EXP-006 v2.8 requires the locked 256-D descriptor interaction")
    if tuple(scalar_indices) != PRIMARY_SCALAR_INDICES:
        raise ValueError("EXP-006 v2.8 requires scalar indices 0-11 and 16-23")
    return list(range(descriptor_dimensions)) + [descriptor_dimensions + index for index in scalar_indices]
