#!/usr/bin/env python3
"""Fast invariant checks for the growing EXP-006 implementation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from revisit3d.experiments import (
    confidence_target,
    fit_confidence_quantiles,
    grouped_folds,
    pose_distillation_loss,
    pose_metrics,
    require_exp006_split,
)
from revisit3d.losses import relative_w2c_from_twist
from revisit3d.models import (
    FeatureMatches,
    PlasticityAtom,
    Sim3Alignment,
    SpatialPlasticityHead,
    backproject_tokens,
    geometry_transport,
    local_knn_scale,
    robust_sim3,
    visual_transport,
)


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

    spatial = SpatialPlasticityHead(feature_dim=32, key_dim=8, code_dim=4, hidden_dim=16)
    features = torch.randn(1, 2, 16, 32)
    base_depth = torch.rand(1, 2, 4, 4) + 0.5
    zero_code = spatial.initial_code(features)
    zero_depth = spatial.depth(features, base_depth, zero_code)
    assert torch.equal(zero_depth, base_depth.flatten(2).unsqueeze(-1))
    updated, _ = spatial.online_update(
        features, base_depth, zero_code,
        lambda depth, code: depth.square().mean() + 1e-4 * code.square().mean(),
        step_size=0.05,
    )
    assert updated.shape == zero_code.shape and updated.abs().sum() > 0
    assert all(parameter.grad is None for parameter in spatial.parameters())

    depth = torch.ones(1, 1, 2, 2)
    intrinsics = torch.tensor([[[2.0, 2.0, 1.0, 1.0]]])
    identity_pose = torch.eye(4).reshape(1, 1, 4, 4)
    points = backproject_tokens(depth, intrinsics, identity_pose, image_size=(2, 2))
    expected = torch.tensor([[[-0.25, -0.25, 1.0], [0.25, -0.25, 1.0],
                              [-0.25, 0.25, 1.0], [0.25, 0.25, 1.0]]])
    assert torch.allclose(points.flatten(1, 2), expected, atol=1e-6)

    generator = torch.Generator().manual_seed(600)
    source_xyz = torch.randn(64, 3, generator=generator)
    angle = torch.tensor(0.4)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
    sim_scale = torch.tensor(1.7)
    translation = torch.tensor([0.4, -0.2, 1.1])
    target_xyz = sim_scale * torch.einsum("ij,nj->ni", rotation, source_xyz) + translation
    matches = FeatureMatches(torch.arange(64), torch.arange(64), torch.full((64,), 0.95))
    alignment = robust_sim3(
        source_xyz, target_xyz, torch.ones(64, 1), torch.ones(64, 1),
        torch.ones(64, 1), matches,
    )
    assert alignment.valid
    assert torch.allclose(alignment.scale, sim_scale, atol=1e-4)
    assert torch.allclose(alignment.rotation, rotation, atol=1e-4)
    assert torch.allclose(alignment.translation, translation, atol=1e-4)
    line = torch.stack((torch.linspace(0, 1, 64), torch.zeros(64), torch.zeros(64)), dim=-1)
    degenerate = robust_sim3(
        line, line, torch.ones(64, 1), torch.ones(64, 1), torch.ones(64, 1), matches,
    )
    assert not degenerate.valid

    source_shape = (1, 1, 64)
    keys = F.normalize(torch.randn(*source_shape, 16, generator=generator), dim=-1)
    code = torch.ones(*source_shape, 4)
    source_points = source_xyz.reshape(*source_shape, 3)
    target_points = target_xyz.reshape(*source_shape, 3)
    source_scale = local_knn_scale(source_points)
    target_scale = local_knn_scale(target_points)
    repeated_views = source_points.repeat(1, 8, 1, 1)
    repeated_scale = local_knn_scale(repeated_views)
    assert torch.allclose(repeated_scale[:, 0], source_scale[:, 0], atol=1e-6)
    assert repeated_scale.min() > 1e-6
    source_atom = PlasticityAtom(source_points, source_scale, keys, code, torch.ones(*source_shape, 1))
    target_atom = PlasticityAtom(target_points, target_scale, keys, torch.zeros_like(code), torch.ones(*source_shape, 1))
    transported = geometry_transport(source_atom, target_atom, [alignment], appearance_weight=5.0)
    assert transported.valid.all() and torch.allclose(transported.code, torch.ones_like(code), atol=1e-5)
    assert (transported.normalized_entropy >= 0).all() and (transported.normalized_entropy <= 1).all()
    visual = visual_transport(source_atom, target_atom)
    assert visual.valid.all() and torch.allclose(visual.code, torch.ones_like(code), atol=1e-5)
    invalid = Sim3Alignment(
        torch.ones(()), torch.eye(3), torch.zeros(3), False, 0, 0, 0.0, float("inf"), 0.0, 0.0,
    )
    rejected = geometry_transport(source_atom, target_atom, [invalid])
    assert not rejected.valid.any() and rejected.code.count_nonzero() == 0

    print("EXP-006 smoke checks passed")


if __name__ == "__main__":
    main()
