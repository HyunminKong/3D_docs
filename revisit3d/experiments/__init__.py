"""Experiment-specific protocol helpers with explicit leakage boundaries."""

from .exp006 import (
    confidence_target,
    deterministic_foreign_indices,
    fit_confidence_quantiles,
    grouped_folds,
    pose_distillation_loss,
    pose_metrics,
    relative_w2c,
    require_exp006_split,
)
from .exp006_atom import (
    CachedAtomSegment,
    CandidateRollout,
    EpisodeRollout,
    adapt_context,
    geometry_objective,
    query_readout_loss,
    run_episode,
    source_to_current,
    symmetric_context_key_loss,
)
from .exp006_router import (
    ALIGNMENT_SCALAR_INDICES,
    DESCRIPTOR_DIMENSIONS,
    OBSERVABLE_SCALAR_DIMENSIONS,
    PRIMARY_SCALAR_INDICES,
    observable_router_features,
    primary_feature_columns,
)

__all__ = [
    "confidence_target",
    "deterministic_foreign_indices",
    "fit_confidence_quantiles",
    "grouped_folds",
    "pose_distillation_loss",
    "pose_metrics",
    "relative_w2c",
    "require_exp006_split",
    "CachedAtomSegment",
    "CandidateRollout",
    "EpisodeRollout",
    "adapt_context",
    "geometry_objective",
    "query_readout_loss",
    "run_episode",
    "source_to_current",
    "symmetric_context_key_loss",
    "ALIGNMENT_SCALAR_INDICES",
    "DESCRIPTOR_DIMENSIONS",
    "OBSERVABLE_SCALAR_DIMENSIONS",
    "PRIMARY_SCALAR_INDICES",
    "observable_router_features",
    "primary_feature_columns",
]
