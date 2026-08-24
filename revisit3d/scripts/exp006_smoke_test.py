#!/usr/bin/env python3
"""Fast invariant checks for the growing EXP-006 implementation."""

from __future__ import annotations

import torch

from revisit3d.experiments import (
    confidence_target,
    fit_confidence_quantiles,
    grouped_folds,
    pose_distillation_loss,
    pose_metrics,
    require_exp006_split,
)
from revisit3d.losses import relative_w2c_from_twist


def _expect_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    require_exp006_split("train")
    require_exp006_split("val", allow_validation=True)
    _expect_error(lambda: require_exp006_split("test", allow_validation=True))

    records = [
        {"source_scene": "a", "target_scene": "b"},
        {"source_scene": "b", "target_scene": "a"},
        {"source_scene": "b", "target_scene": "c"},
        {"source_scene": "d", "target_scene": "e"},
        {"source_scene": "f", "target_scene": "g"},
    ]
    folds, groups = grouped_folds(records, folds=3, seed=600)
    assert len({groups[index] for index in (0, 1, 2)}) == 1
    assert len({index for fold in folds for index in fold}) == len(records)
    group_fold = {}
    for fold, indices in enumerate(folds):
        for index in indices:
            previous = group_fold.setdefault(groups[index], fold)
            assert previous == fold

    latent = torch.linspace(-8, 2, 100)
    raw_confidence = 1 + latent.exp()
    q_low, q_high = fit_confidence_quantiles([raw_confidence])
    target = confidence_target(raw_confidence, q_low, q_high)
    assert target.min() == 0 and target.max() == 1
    assert torch.all(target[1:] >= target[:-1])
    assert target.std() > 0.1

    twist = torch.zeros(2, 4, 6)
    twist[..., 0] = torch.tensor([0.0, 0.2, 0.4, 0.6])
    twist[..., 5] = torch.tensor([0.0, 0.02, 0.04, 0.06])
    target_w2c = relative_w2c_from_twist(twist)
    loss, terms = pose_distillation_loss(twist, target_w2c, motion_threshold=1e-6)
    metrics = pose_metrics(twist, target_w2c, motion_threshold=1e-6)
    assert loss < 1e-5, (loss, terms)
    assert metrics["rotation_error_deg"].max() < 1e-4
    assert metrics["translation_direction_error_deg"].max() < 1e-3
    assert metrics["scale_aligned_translation_error"] < 1e-6
    assert metrics["view0_identity_error"] < 1e-6

    print("EXP-006 smoke checks passed")


if __name__ == "__main__":
    main()
