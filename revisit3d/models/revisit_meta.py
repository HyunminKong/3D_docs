"""Memory-free, revisit-aware meta-learning for compact TTT coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry_head import CompactTTTState, StreamingGeometryHead


class SignedResidualTransport(nn.Module):
    """Map a prior adaptation and current context to a signed residual.

    This is deliberately not a bank reader.  During this phase the matched A
    state is supplied by the benchmark oracle.  The module answers whether a
    reusable coordinate can be learned before retrieval is introduced.
    """

    def __init__(self, feature_dim: int, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, state_dim), nn.Tanh())
        self.residual = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim)
        )
        # A zero residual is the valid cold/current-TTT baseline at step zero.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, prior: CompactTTTState, query_features: Tensor) -> CompactTTTState:
        query_context = query_features.mean(dim=(1, 2))
        context = self.context(query_context)
        return CompactTTTState(self.residual(torch.cat((prior.value, context), dim=-1)))


@dataclass
class RevisitRollout:
    state_a: CompactTTTState
    state_ab: CompactTTTState
    state_cold: CompactTTTState
    state_reuse: CompactTTTState
    prediction_cold: dict[str, Tensor]
    prediction_reuse: dict[str, Tensor]


class RevisitMetaLearner(nn.Module):
    """Unroll A→B→A' without an adaptation memory implementation."""

    def __init__(self, head: StreamingGeometryHead, transport: SignedResidualTransport):
        super().__init__()
        self.head = head
        self.transport = transport

    def rollout(
        self,
        features_a: Tensor,
        features_b: Tensor,
        features_a_prime: Tensor,
        online_objective: Callable[[dict[str, Tensor], str], Tensor],
        *,
        features_a_prime_query: Tensor | None = None,
        steps: int = 1,
        learning_rate: float = 1e-2,
        create_graph: bool = True,
        retain_state_gradient: bool = True,
    ) -> RevisitRollout:
        initial = self.head.initial_state(features_a.shape[0], device=features_a.device, dtype=features_a.dtype)
        adapt = lambda features, state, tag: self.head.adapt(
            features, state, lambda prediction: online_objective(prediction, tag),
            steps=steps, learning_rate=learning_rate, create_graph=create_graph,
            retain_state_gradient=retain_state_gradient,
        )[0]
        state_a = adapt(features_a, initial, "a")
        state_ab = adapt(features_b, state_a, "b")
        state_cold = adapt(features_a_prime, initial, "a_prime")
        prior = self.transport(state_a, features_a_prime)
        state_reuse = adapt(features_a_prime, CompactTTTState(state_ab.value + prior.value), "a_prime")
        query_features = features_a_prime if features_a_prime_query is None else features_a_prime_query
        return RevisitRollout(
            state_a=state_a, state_ab=state_ab, state_cold=state_cold, state_reuse=state_reuse,
            prediction_cold=self.head(query_features, state_cold),
            prediction_reuse=self.head(query_features, state_reuse),
        )

    @staticmethod
    def revisit_outer_loss(
        rollout: RevisitRollout,
        heldout_objective: Callable[[dict[str, Tensor]], Tensor],
        *,
        margin: float = 0.0,
    ) -> Tensor:
        """Optimise reuse quality and penalise reuse that loses to cold TTT."""
        reuse = heldout_objective(rollout.prediction_reuse)
        cold = heldout_objective(rollout.prediction_cold)
        return reuse + F.softplus(reuse - cold + margin)
