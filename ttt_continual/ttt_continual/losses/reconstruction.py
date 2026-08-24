"""What a rendered chunk is scored on.

L2 dominates and sets the scale. SSIM is added because L2 alone is happy to
produce a blurred average of a moving scene, which is exactly the failure a
dynamic benchmark should punish. Opacity is pushed down so that primitives which
explain nothing fade instead of accumulating as fog, and depth is regularised
only weakly -- the ray parameterisation already prevents the worst of it.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..metrics.image import ssim


@dataclass
class LossWeights:
    l2: float = 1.0
    ssim: float = 0.2
    opacity: float = 0.01
    distortion: float = 0.0


def reconstruction_loss(render: Dict[str, torch.Tensor], target_rgb: torch.Tensor,
                        gaussians: Dict[str, torch.Tensor], weights: LossWeights,
                        mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    pred = render["rgb"]
    if mask is not None:
        m = mask.float().unsqueeze(1)
        l2 = ((pred - target_rgb) ** 2 * m).sum() / m.sum().clamp(min=1.0) / pred.shape[1]
    else:
        l2 = F.mse_loss(pred, target_rgb)

    terms = {"l2": l2}
    total = weights.l2 * l2

    if weights.ssim > 0:
        d = 1.0 - ssim(pred.clamp(0, 1), target_rgb).mean()
        terms["ssim"] = d
        total = total + weights.ssim * d

    if weights.opacity > 0:
        o = gaussians["opacity"].sigmoid().mean()
        terms["opacity"] = o
        total = total + weights.opacity * o

    if weights.distortion > 0:
        depth = gaussians["depth"]
        d = depth.diff(dim=1).abs().mean() if depth.shape[1] > 1 else depth.sum() * 0
        terms["distortion"] = d
        total = total + weights.distortion * d

    terms["total"] = total
    return terms
