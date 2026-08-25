"""Paper-minimal local plasticity objective with explicit future boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from revisit3d.experiments.exp006_atom import CachedAtomSegment
from revisit3d.losses import track_3d_consistency_loss
from revisit3d.models import PlasticityAtom, SpatialPlasticityHead, visual_transport


@dataclass
class MinimalCurrentState:
    zero: PlasticityAtom
    query_zero: PlasticityAtom
    current: PlasticityAtom
    base_query_loss: Tensor
    current_query_loss: Tensor


@dataclass
class MinimalEpisode:
    outer_loss: Tensor
    base_query_loss: Tensor
    current_query_loss: Tensor
    reuse_query_loss: Tensor
    utility: Tensor


def track_objective(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    code: Tensor,
) -> Tensor:
    """The single online/meta geometry loss selected by EXP-011."""
    depth = head.depth(segment.features, segment.base_depth, code)
    points = depth.shape[2]
    side = int(points ** 0.5)
    if side * side != points:
        raise ValueError("depth tokens must form a square grid")
    return track_3d_consistency_loss(
        depth.squeeze(-1).reshape(depth.shape[0], depth.shape[1], side, side),
        segment.intrinsics,
        segment.predicted_w2c,
        segment.track,
        segment.track_visibility,
        segment.track_confidence,
        image_size=segment.image_size,
    )


def adapt_minimal(
    head: SpatialPlasticityHead,
    segment: CachedAtomSegment,
    initial_code: Tensor,
    *,
    step_size: float,
) -> Tensor:
    if segment.role == "query":
        raise ValueError("future query cannot enter online adaptation")
    code, _ = head.online_update(
        segment.features,
        segment.base_depth,
        initial_code,
        lambda _depth, state: track_objective(head, segment, state),
        step_size=step_size,
        steps=1,
    )
    return code


def future_readout(
    head: SpatialPlasticityHead,
    context: PlasticityAtom,
    query: CachedAtomSegment,
    query_zero: PlasticityAtom,
) -> Tensor:
    """Offline meta-label readout; query evidence never changes the context."""
    if query.role != "query":
        raise ValueError("future readout requires a query-role segment")
    query_code = visual_transport(context, query_zero).code
    return track_objective(head, query, query_code)


def prepare_current(
    head: SpatialPlasticityHead,
    current: CachedAtomSegment,
    query: CachedAtomSegment,
    *,
    step_size: float,
) -> MinimalCurrentState:
    if current.role != "current" or query.role != "query":
        raise ValueError("current/query role boundary violated")
    zero = current.atom(head)
    query_zero = query.atom(head)
    base_query_loss = track_objective(head, query, query_zero.code)
    current_code = adapt_minimal(head, current, zero.code, step_size=step_size)
    current_atom = replace(zero, code=current_code)
    current_query_loss = future_readout(head, current_atom, query, query_zero)
    return MinimalCurrentState(
        zero=zero,
        query_zero=query_zero,
        current=current_atom,
        base_query_loss=base_query_loss,
        current_query_loss=current_query_loss,
    )


def reuse_query_loss(
    head: SpatialPlasticityHead,
    source: CachedAtomSegment,
    current: CachedAtomSegment,
    query: CachedAtomSegment,
    state: MinimalCurrentState,
    *,
    step_size: float,
    reuse_strength: float,
) -> Tensor:
    if source.role != "source":
        raise ValueError("memory source must have source role")
    source_zero = source.atom(head)
    source_code = adapt_minimal(head, source, source_zero.code, step_size=step_size)
    source_atom = replace(source_zero, code=source_code)
    transported = visual_transport(source_atom, state.zero).code
    reused_code = (state.current.code + reuse_strength * transported).clamp(-1, 1)
    reused = replace(state.zero, code=reused_code)
    return future_readout(head, reused, query, state.query_zero)


def run_minimal_episode(
    head: SpatialPlasticityHead,
    source: CachedAtomSegment,
    current: CachedAtomSegment,
    query: CachedAtomSegment,
    *,
    step_size: float,
    reuse_strength: float,
) -> MinimalEpisode:
    state = prepare_current(head, current, query, step_size=step_size)
    reused = reuse_query_loss(
        head, source, current, query, state,
        step_size=step_size, reuse_strength=reuse_strength,
    )
    denominator = state.base_query_loss.detach().abs().clamp_min(1e-6)
    # One meta-objective, with an equal current/reuse average and no auxiliary
    # key, neutralization, centering, smoothness, or code-norm terms.
    outer = 0.5 * (state.current_query_loss + reused) / denominator
    utility = (
        state.current_query_loss.detach() - reused.detach()
    ) / state.current_query_loss.detach().abs().clamp_min(1e-6)
    return MinimalEpisode(
        outer_loss=outer,
        base_query_loss=state.base_query_loss,
        current_query_loss=state.current_query_loss,
        reuse_query_loss=reused,
        utility=utility,
    )


__all__ = [
    "MinimalCurrentState", "MinimalEpisode", "adapt_minimal", "future_readout",
    "prepare_current", "reuse_query_loss", "run_minimal_episode", "track_objective",
]
