"""Leakage-safe Stage-1 atom rollouts for EXP-006."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from revisit3d.losses import (
    depth_smoothness_loss,
    normalized_future_utility,
    track_3d_consistency_loss,
    utility_masks,
)
from revisit3d.models import (
    PlasticityAtom,
    SpatialPlasticityHead,
    align_atoms,
    build_plasticity_atom,
    geometry_transport,
    visual_transport,
)


Role = Literal["source", "current", "query"]


@dataclass
class CachedAtomSegment:
    role: Role
    scene: str
    features: Tensor
    rgb: Tensor
    intrinsics: Tensor
    base_depth: Tensor
    base_confidence: Tensor
    predicted_w2c: Tensor
    xyz: Tensor
    scale: Tensor
    track: Tensor
    track_visibility: Tensor
    track_confidence: Tensor
    image_size: tuple[int, int]

    @classmethod
    def from_cache(
        cls, payload: dict, role: Role, device: torch.device, *, feature_dtype: torch.dtype = torch.float32,
    ) -> "CachedAtomSegment":
        if role not in ("source", "current", "query"):
            raise ValueError(f"invalid segment role {role!r}")
        return cls(
            role=role,
            scene=payload["scene"],
            features=payload["features"].to(device=device, dtype=feature_dtype),
            rgb=payload["rgb_uint8"].to(device=device, dtype=torch.float32).div_(255),
            intrinsics=payload["intrinsics"].to(device=device, dtype=torch.float32),
            base_depth=payload["base_depth"].to(device=device, dtype=torch.float32),
            base_confidence=payload["base_confidence"].to(device=device, dtype=torch.float32),
            predicted_w2c=payload["predicted_w2c"].to(device=device, dtype=torch.float32),
            xyz=payload["xyz"].to(device=device, dtype=torch.float32),
            scale=payload["scale"].to(device=device, dtype=torch.float32),
            track=payload["track"].to(device=device, dtype=torch.float32),
            track_visibility=payload["track_visibility"].to(device=device, dtype=torch.float32),
            track_confidence=payload["track_confidence"].to(device=device, dtype=torch.float32),
            image_size=tuple(payload["image_size"]),
        )

    def atom(self, head: SpatialPlasticityHead, code: Tensor | None = None) -> PlasticityAtom:
        return build_plasticity_atom(
            head, self.features, self.xyz, self.scale, self.base_confidence,
            self.track, self.track_visibility, self.track_confidence,
            image_size=self.image_size, code=code,
        )


@dataclass
class CandidateRollout:
    label: str
    valid: bool
    alignment_valid: bool
    source_atom: PlasticityAtom
    current_atom: PlasticityAtom
    query_loss: Tensor
    pre_update_residual: Tensor
    current_pre_loss: Tensor
    current_post_loss: Tensor
    alignment_inlier_ratio: float
    alignment_residual: float


@dataclass
class EpisodeRollout:
    outer_loss: Tensor
    base_query_loss: Tensor
    current_query_loss: Tensor
    candidates: list[CandidateRollout]
    utilities: Tensor
    valid: Tensor
    key_loss: Tensor
    benefit_loss: Tensor
    neutral_loss: Tensor
    center_loss: Tensor
    counts: dict[str, int]


def geometry_objective(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    code: Tensor,
    *,
    smoothness_weight: float = 1e-3,
    code_weight: float = 1e-4,
    return_stats: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    depth = head.depth(segment.features, segment.base_depth, code)
    points = depth.shape[2]
    side = int(points ** 0.5)
    if side * side != points:
        raise ValueError("depth tokens must form a square grid")
    depth_grid = depth.squeeze(-1).reshape(depth.shape[0], depth.shape[1], side, side)
    track, stats = track_3d_consistency_loss(
        depth_grid, segment.intrinsics, segment.predicted_w2c, segment.track,
        segment.track_visibility, segment.track_confidence,
        image_size=segment.image_size, return_stats=True,
    )
    smooth = depth_smoothness_loss(depth_grid, segment.rgb)
    regularizer = code.square().mean()
    total = track + smoothness_weight * smooth + code_weight * regularizer
    if return_stats:
        return total, {**stats, "track_loss": track, "smoothness": smooth, "code_l2": regularizer}
    return total


def adapt_context(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    initial_code: Tensor,
    *,
    step_size: float,
    steps: int = 1,
    retain_initial_gradient: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    if segment.role == "query":
        raise ValueError("future query is read-only and cannot enter online adaptation")
    return head.online_update(
        segment.features, segment.base_depth, initial_code,
        lambda _depth, code: geometry_objective(head, segment, code),
        step_size=step_size, steps=steps, retain_initial_gradient=retain_initial_gradient,
    )


def source_to_current(
    source: PlasticityAtom,
    current: PlasticityAtom,
    *,
    source_role: Role,
    current_role: Role,
    appearance_weight: float,
) -> tuple[Tensor, object]:
    if source_role == "query" or current_role == "query":
        raise ValueError("query geometry cannot enter Sim(3) alignment")
    detached_source = replace(source, key=source.key.detach(), code=source.code.detach())
    detached_current = replace(current, key=current.key.detach(), code=current.code.detach())
    alignment = align_atoms(detached_source, detached_current)[0]
    transported = geometry_transport(source, current, [alignment], appearance_weight=appearance_weight)
    return transported.code, alignment


def query_readout_loss(
    head: SpatialPlasticityHead,
    context_atom: PlasticityAtom,
    query: CachedAtomSegment,
) -> Tensor:
    """Read a context fast state at query tokens without query geometry."""
    if query.role != "query":
        raise ValueError("query readout requires a query-role segment")
    query_atom = query.atom(head)
    query_code = visual_transport(context_atom, query_atom).code
    return geometry_objective(head, query, query_code)


def symmetric_context_key_loss(
    head: SpatialPlasticityHead,
    source: CachedAtomSegment,
    *,
    target_view: int,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE with positives mined from detached, unprojected context features."""
    if source.role != "source":
        raise ValueError("key positives must come from a source context")
    views = source.features.shape[1]
    if not 1 <= target_view < views:
        raise ValueError("target_view must be a non-reference context view")
    raw_left = F.normalize(source.features[:, 0].detach(), dim=-1)
    raw_right = F.normalize(source.features[:, target_view].detach(), dim=-1)
    raw_similarity = raw_left @ raw_right.transpose(-1, -2)
    right_for_left = raw_similarity.argmax(dim=-1)
    left_for_right = raw_similarity.argmax(dim=-2)
    left_index = torch.arange(raw_left.shape[1], device=raw_left.device)
    mutual = left_for_right.gather(1, right_for_left) == left_index[None]
    selected_left = left_index[mutual[0]].detach()
    selected_right = right_for_left[0, selected_left].detach()
    if selected_left.numel() == 0:
        return head.key_projection.weight.sum() * 0
    key = head.appearance_key(source.features)
    left, right = key[0, 0], key[0, target_view]
    logits_lr = left[selected_left] @ right.transpose(0, 1) / temperature
    logits_rl = right[selected_right] @ left.transpose(0, 1) / temperature
    return 0.5 * (
        F.cross_entropy(logits_lr, selected_right)
        + F.cross_entropy(logits_rl, selected_left)
    )


def run_episode(
    head: SpatialPlasticityHead,
    current: CachedAtomSegment,
    query: CachedAtomSegment,
    sources: list[tuple[str, CachedAtomSegment]],
    *,
    step_size: float,
    ttt_steps: int,
    appearance_weight: float,
    reuse_strength: float,
    utility_epsilon: float,
    key_temperature: float,
    key_target_view: int,
    transport_mode: Literal["geometry_appearance", "visual"] = "geometry_appearance",
    build_outer: bool = True,
) -> EpisodeRollout:
    if current.role != "current" or query.role != "query":
        raise ValueError("run_episode requires current and read-only query roles")
    current_zero = current.atom(head)
    base_query_loss = query_readout_loss(head, current_zero, query)
    current_code, _ = adapt_context(
        head, current, current_zero.code, step_size=step_size, steps=ttt_steps,
    )
    current_atom = replace(current_zero, code=current_code)
    current_query_loss = query_readout_loss(head, current_atom, query)
    candidates: list[CandidateRollout] = []
    for label, source_segment in sources:
        source_zero = source_segment.atom(head)
        source_code, _ = adapt_context(
            head, source_segment, source_zero.code, step_size=step_size, steps=ttt_steps,
        )
        source_atom = replace(source_zero, code=source_code.detach())
        if transport_mode == "geometry_appearance":
            transported, alignment = source_to_current(
                source_atom, current_zero, source_role=source_segment.role, current_role=current.role,
                appearance_weight=appearance_weight,
            )
            candidate_valid = alignment.valid
        elif transport_mode == "visual":
            # Geometry remains observable router evidence, but is not the fast-code
            # carrier and cannot hard-reject a visually addressable memory.
            alignment = align_atoms(
                replace(source_atom, key=source_atom.key.detach(), code=source_atom.code.detach()),
                replace(current_zero, key=current_zero.key.detach(), code=current_zero.code.detach()),
            )[0]
            visual = visual_transport(source_atom, current_zero)
            transported = visual.code
            candidate_valid = bool(visual.valid.all())
        else:
            raise ValueError(f"unknown transport_mode {transport_mode!r}")
        if candidate_valid:
            pre_loss = geometry_objective(head, current, current_code)
            adapted_code = (current_code.detach() + reuse_strength * transported).clamp(-1, 1)
            pre_residual = (
                head.log_depth_residual(current.features, adapted_code)
                - head.log_depth_residual(current.features, current_code.detach())
            )
            candidate_atom = replace(current_zero, code=adapted_code)
            post_loss = geometry_objective(head, current, adapted_code)
            query_loss = query_readout_loss(head, candidate_atom, query)
        else:
            pre_loss = current_query_loss.detach()
            post_loss = current_query_loss.detach()
            pre_residual = transported.sum() * torch.zeros((), device=transported.device)
            candidate_atom = replace(current_zero, code=current_code.detach())
            query_loss = current_query_loss.detach()
        candidates.append(CandidateRollout(
            label=label, valid=candidate_valid, alignment_valid=alignment.valid, source_atom=source_atom,
            current_atom=candidate_atom, query_loss=query_loss,
            pre_update_residual=pre_residual, current_pre_loss=pre_loss,
            current_post_loss=post_loss, alignment_inlier_ratio=alignment.inlier_ratio,
            alignment_residual=alignment.normalized_median_residual,
        ))
    candidate_losses = torch.stack([candidate.query_loss for candidate in candidates])
    valid = torch.tensor([candidate.valid for candidate in candidates], device=candidate_losses.device)
    utilities = normalized_future_utility(current_query_loss, candidate_losses)
    masks = utility_masks(utilities.detach(), valid, utility_epsilon)
    selected = masks["beneficial"].clone()
    if not selected.any() and valid.any():
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        best = valid_indices[utilities.detach()[valid].argmax()]
        selected[best] = True
    denominator = base_query_loss.detach().abs().clamp_min(1e-6)
    ell_current = current_query_loss / denominator
    if selected.any():
        weights = torch.softmax(utilities.detach()[selected] / 0.10, dim=0)
        benefit = (weights * candidate_losses[selected] / denominator).sum()
    else:
        benefit = ell_current
    harmful_unselected = masks["harmful"] & ~selected
    if harmful_unselected.any():
        neutral = torch.stack([
            candidate.pre_update_residual.abs().mean()
            for index, candidate in enumerate(candidates) if bool(harmful_unselected[index])
        ]).mean()
    else:
        neutral = current_query_loss.new_zeros(())
    source_centers = torch.stack([
        head.log_depth_residual(source_segment.features, candidate.source_atom.code).mean().square()
        for candidate, (_, source_segment) in zip(candidates, sources)
    ])
    center = source_centers.mean()
    key_loss = symmetric_context_key_loss(
        head, sources[0][1], target_view=key_target_view, temperature=key_temperature,
    )
    # The relative margin treats current-only as a detached reference.  Without
    # this boundary, minimizing softplus(benefit-current) can increase the
    # current loss and manufacture large relative utility (preserved v2.4
    # failure).  The explicit current term protects absolute TTT quality.
    outer = (
        ell_current + benefit + F.softplus(benefit - ell_current.detach())
        + 0.10 * neutral + 0.01 * center + 0.05 * key_loss
    )
    if not build_outer:
        outer = outer.detach()
    return EpisodeRollout(
        outer_loss=outer, base_query_loss=base_query_loss, current_query_loss=current_query_loss,
        candidates=candidates, utilities=utilities, valid=valid, key_loss=key_loss,
        benefit_loss=benefit, neutral_loss=neutral, center_loss=center,
        counts={name: int(mask.sum()) for name, mask in masks.items()},
    )


__all__ = [
    "CachedAtomSegment", "CandidateRollout", "EpisodeRollout", "adapt_context",
    "geometry_objective", "query_readout_loss", "run_episode", "source_to_current",
    "symmetric_context_key_loss",
]
