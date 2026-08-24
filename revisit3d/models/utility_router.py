"""Observable utility/risk routing for local plasticity memories.

The router never receives a future/query quantity.  Predicted geometry is an
evidence channel (alignment validity, inlier ratio, residual, and coverage),
not the carrier of the fast code.  The memory code is transported visually and
is only applied as a bounded residual after the current-context TTT step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class UtilityRiskPrediction:
    utility: Tensor
    risk_logit: Tensor


@dataclass(frozen=True)
class RoutingDecision:
    selected_index: Tensor
    accepted: Tensor
    weights: Tensor


class ObservableUtilityRiskRouter(nn.Module):
    """Predict future utility using only current/source observable evidence.

    Args:
        descriptor_dim: Dimension of the pooled local appearance keys.
        scalar_dim: Number of normalized online/transport/geometry statistics.
        projected_dim: Shared descriptor projection size.

    Inputs are current and candidate descriptors plus scalar measurements.  A
    geometry alignment failure must be represented in ``observable_scalars``;
    it is deliberately not an availability mask because visual transport is
    still defined when predicted 3D alignment fails.
    """

    def __init__(
        self,
        descriptor_dim: int = 64,
        scalar_dim: int = 16,
        projected_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.scalar_dim = scalar_dim
        self.projected_dim = projected_dim
        self.descriptor_projection = nn.Sequential(
            nn.Linear(descriptor_dim, projected_dim),
            nn.GELU(),
            nn.LayerNorm(projected_dim),
        )
        input_dim = 4 * projected_dim + scalar_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def build_features(
        self,
        current_descriptor: Tensor,
        candidate_descriptor: Tensor,
        observable_scalars: Tensor,
    ) -> Tensor:
        if current_descriptor.ndim != 2 or current_descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError("current_descriptor must be [batch, descriptor_dim]")
        if candidate_descriptor.ndim != 3 or candidate_descriptor.shape[-1] != self.descriptor_dim:
            raise ValueError("candidate_descriptor must be [batch, candidates, descriptor_dim]")
        if observable_scalars.ndim != 3 or observable_scalars.shape[-1] != self.scalar_dim:
            raise ValueError("observable_scalars must be [batch, candidates, scalar_dim]")
        if candidate_descriptor.shape[:2] != observable_scalars.shape[:2]:
            raise ValueError("candidate descriptors and scalar statistics must share [batch, candidates]")
        if current_descriptor.shape[0] != candidate_descriptor.shape[0]:
            raise ValueError("current and candidate descriptors must share the batch dimension")

        current = self.descriptor_projection(current_descriptor).unsqueeze(1)
        current = current.expand(-1, candidate_descriptor.shape[1], -1)
        candidate = self.descriptor_projection(candidate_descriptor)
        return torch.cat(
            (current, candidate, current - candidate, current * candidate, observable_scalars), dim=-1,
        )

    def forward(
        self,
        current_descriptor: Tensor,
        candidate_descriptor: Tensor,
        observable_scalars: Tensor,
    ) -> UtilityRiskPrediction:
        prediction = self.head(self.build_features(
            current_descriptor, candidate_descriptor, observable_scalars,
        ))
        return UtilityRiskPrediction(utility=prediction[..., 0], risk_logit=prediction[..., 1])

    @staticmethod
    def hard_route(
        prediction: UtilityRiskPrediction,
        *,
        utility_threshold: float = 0.0,
        risk_threshold: float = 0.5,
        candidate_available: Tensor | None = None,
    ) -> RoutingDecision:
        """Select at most one candidate, with current-only TTT as rejection."""
        if prediction.utility.shape != prediction.risk_logit.shape or prediction.utility.ndim != 2:
            raise ValueError("utility and risk logits must have shape [batch, candidates]")
        if candidate_available is None:
            candidate_available = torch.ones_like(prediction.utility, dtype=torch.bool)
        elif candidate_available.shape != prediction.utility.shape:
            raise ValueError("candidate_available must match prediction shape")
        eligible = (
            candidate_available.bool()
            & (prediction.utility > utility_threshold)
            & (prediction.risk_logit.sigmoid() < risk_threshold)
        )
        score = prediction.utility.masked_fill(~eligible, torch.finfo(prediction.utility.dtype).min)
        selected = score.argmax(dim=-1)
        accepted = eligible.any(dim=-1)
        selected = torch.where(accepted, selected, torch.full_like(selected, -1))
        weights = torch.zeros_like(prediction.utility)
        if accepted.any():
            rows = accepted.nonzero(as_tuple=False).flatten()
            weights[rows, selected[rows]] = 1
        return RoutingDecision(selected_index=selected, accepted=accepted, weights=weights)


def apply_bounded_memory_residual(
    current_code: Tensor,
    transported_codes: Tensor,
    weights: Tensor,
    *,
    strength: float = 0.10,
    clamp: tuple[float, float] = (-1.0, 1.0),
) -> Tensor:
    """Apply routed visual memories after exactly one current-context TTT step."""
    if current_code.ndim < 2 or transported_codes.ndim != current_code.ndim + 1:
        raise ValueError("transported_codes must insert a candidate axis after batch")
    if transported_codes.shape[0] != current_code.shape[0] or transported_codes.shape[2:] != current_code.shape[1:]:
        raise ValueError("transported code shape does not match current code")
    if weights.shape != transported_codes.shape[:2]:
        raise ValueError("weights must be [batch, candidates]")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    broadcast = weights.reshape(*weights.shape, *([1] * (current_code.ndim - 1)))
    memory = (transported_codes * broadcast).sum(dim=1)
    return (current_code + strength * memory).clamp(*clamp)


__all__ = [
    "ObservableUtilityRiskRouter", "RoutingDecision", "UtilityRiskPrediction",
    "apply_bounded_memory_residual",
]
