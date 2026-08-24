"""Future-utility targets and losses for EXP-006.

Future losses are outer supervision only.  Nothing in this module is an
online signal or a router input.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def normalized_future_utility(current_loss: Tensor, candidate_loss: Tensor) -> Tensor:
    """Positive means that a candidate improves over current-only TTT."""
    denominator = current_loss.detach().abs().clamp_min(1e-6)
    return (current_loss - candidate_loss) / denominator


def utility_masks(utility: Tensor, valid: Tensor, epsilon: float) -> dict[str, Tensor]:
    if utility.shape != valid.shape:
        raise ValueError("utility and valid masks must have identical shapes")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    valid = valid.bool()
    return {
        "beneficial": valid & (utility > epsilon),
        "neutral": valid & (utility.abs() <= epsilon),
        "harmful": valid & (utility < -epsilon),
    }


def utility_risk_loss(
    utility_hat: Tensor,
    risk_logit: Tensor,
    utility_target: Tensor,
    valid: Tensor,
    *,
    epsilon: float,
    positive_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Masked Stage-2 objective; neutral and invalid candidates have no BCE."""
    if not (utility_hat.shape == risk_logit.shape == utility_target.shape == valid.shape):
        raise ValueError("all utility/risk tensors must have identical shapes")
    valid = valid.bool()
    regression = F.smooth_l1_loss(
        utility_hat[valid], utility_target.detach()[valid], reduction="mean",
    ) if valid.any() else utility_hat.sum() * 0
    nonneutral = valid & (utility_target.detach().abs() > epsilon)
    risk_target = (utility_target.detach() < -epsilon).to(risk_logit.dtype)
    risk = F.binary_cross_entropy_with_logits(
        risk_logit[nonneutral], risk_target[nonneutral], pos_weight=positive_weight,
    ) if nonneutral.any() else risk_logit.sum() * 0
    return regression + risk, {"utility": regression, "risk": risk, "risk_mask": nonneutral}


def future_regret(selected_loss: Tensor, current_loss: Tensor, candidate_losses: Tensor) -> Tensor:
    oracle = torch.minimum(current_loss, candidate_losses.min(dim=-1).values)
    return selected_loss - oracle


__all__ = [
    "normalized_future_utility", "utility_masks", "utility_risk_loss", "future_regret",
]
