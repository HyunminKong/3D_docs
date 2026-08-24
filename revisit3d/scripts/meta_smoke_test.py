#!/usr/bin/env python3
"""Unit/smoke tests for online reprojection loss and revisit meta gradients."""

import torch

from revisit3d.losses import relative_w2c_from_twist, reprojection_loss
from revisit3d.models import RevisitMetaLearner, SignedResidualTransport, StreamingGeometryHead


def main() -> None:
    torch.manual_seed(0)
    image = torch.rand(1, 1, 3, 8, 8)
    images = image.repeat(1, 2, 1, 1, 1)
    depth = torch.ones(1, 2, 4, 4)
    intrinsics = torch.tensor([[[4.0, 4.0, 1.5, 1.5], [4.0, 4.0, 1.5, 1.5]]])
    w2c = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    assert reprojection_loss(depth, images, intrinsics, w2c).item() < 1e-5
    assert torch.allclose(relative_w2c_from_twist(torch.zeros(1, 2, 6)), w2c)

    head = StreamingGeometryHead(feature_dim=32, state_dim=8, hidden_dim=48)
    transport = SignedResidualTransport(feature_dim=32, state_dim=8, hidden_dim=32)
    learner = RevisitMetaLearner(head, transport)
    features = [torch.randn(2, 2, 9, 32) for _ in range(3)]
    rollout = learner.rollout(*features, features_a_prime_query=features[-1],
                              online_objective=lambda prediction, _: prediction["depth"].mean())
    loss = learner.revisit_outer_loss(rollout, lambda prediction: prediction["pointmap"].square().mean())
    loss.backward()
    assert any(parameter.grad is not None for parameter in transport.parameters())
    print(f"reprojection={reprojection_loss(depth, images, intrinsics, w2c).item():.2e} | meta loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
