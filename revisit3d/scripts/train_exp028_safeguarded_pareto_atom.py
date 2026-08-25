#!/usr/bin/env python3
"""EXP-028: preserve common descent after the offline AdamW proposal."""

from __future__ import annotations

import torch

from revisit3d.scripts import train_exp027_pareto_plasticity_atom as protocol


def _safe_optimizer_step(
    parameters, optimizer, common_gradient, log_unit, relative_unit, *, epsilon, tolerance
):
    protocol._assign_flat_gradient(parameters, common_gradient)
    before = protocol._flat_parameters(parameters)
    optimizer.step()
    proposed = protocol._flat_parameters(parameters) - before
    descent = -proposed
    norm = descent.norm().clamp_min(epsilon)
    log_margin = torch.dot(log_unit, descent / norm)
    relative_margin = torch.dot(relative_unit, descent / norm)
    safeguard = bool(log_margin <= tolerance or relative_margin <= tolerance)
    if safeguard:
        safe_direction = common_gradient / common_gradient.norm().clamp_min(epsilon)
        safe_displacement = -proposed.norm() * safe_direction
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                count = parameter.numel()
                parameter.copy_(
                    (before[offset:offset + count] + safe_displacement[offset:offset + count])
                    .reshape_as(parameter)
                )
                offset += count
        descent = -safe_displacement
        norm = descent.norm().clamp_min(epsilon)
        log_margin = torch.dot(log_unit, descent / norm)
        relative_margin = torch.dot(relative_unit, descent / norm)
    if log_margin <= tolerance or relative_margin <= tolerance:
        raise RuntimeError("EXP-028 safeguard failed to preserve common descent")
    return {
        "realized_log_margin": log_margin,
        "realized_relative_margin": relative_margin,
        "safeguard_applied": safeguard,
    }


if __name__ == "__main__":
    protocol._optimizer_step = _safe_optimizer_step
    protocol.main()
