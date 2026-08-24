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

__all__ = [
    "confidence_target",
    "deterministic_foreign_indices",
    "fit_confidence_quantiles",
    "grouped_folds",
    "pose_distillation_loss",
    "pose_metrics",
    "relative_w2c",
    "require_exp006_split",
]
