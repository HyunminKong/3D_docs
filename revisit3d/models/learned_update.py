"""First-order, feature-conditioned update rule for compact anchored states."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LocalTrackUpdateRule(nn.Module):
    """Map local TTT gradient and frozen slot context to a bounded state step."""

    def __init__(self, feature_dim: int, context_dim: int = 32, hidden_dim: int = 64, step_scale: float = 0.1):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, context_dim), nn.GELU())
        self.update = nn.Sequential(nn.Linear(context_dim + 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.step_scale = step_scale

    def forward(self, state: Tensor, gradient: Tensor, slot_features: Tensor) -> Tensor:
        if state.shape != gradient.shape or slot_features.shape[:2] != state.shape:
            raise ValueError("state/gradient/slot feature dimensions are inconsistent")
        inputs = torch.cat((state.unsqueeze(-1), gradient.unsqueeze(-1), self.context(slot_features)), dim=-1)
        return self.step_scale * torch.tanh(self.update(inputs).squeeze(-1))
