"""Image metrics, with the masked variants the driving data needs.

PSNR here is computed per view and then averaged, not computed once over a
pooled error. The two differ whenever views vary in difficulty, and the per-view
form is the one that answers "how well is each viewpoint reconstructed", which
is the question every measurement in this project asks.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor,
         mask: Optional[torch.Tensor] = None, eps: float = 1e-10) -> torch.Tensor:
    """(v, c, h, w) -> (v,). `mask` is True where a pixel counts."""
    err = (pred.float() - target.float()) ** 2
    if mask is None:
        mse = err.flatten(1).mean(1)
    else:
        m = mask.float().unsqueeze(1).expand_as(err)
        mse = (err * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1.0)
    return -10.0 * torch.log10(mse.clamp(min=eps))


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11,
         sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """Structural similarity, per view. Standard constants."""
    c = pred.shape[1]
    w = _gaussian_window(window, sigma, pred.device).expand(c, 1, window, window)
    pad = window // 2
    mu_p = F.conv2d(pred, w, padding=pad, groups=c)
    mu_t = F.conv2d(target, w, padding=pad, groups=c)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    var_p = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu_p2
    var_t = F.conv2d(target * target, w, padding=pad, groups=c) - mu_t2
    cov = F.conv2d(pred * target, w, padding=pad, groups=c) - mu_pt
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_pt + c1) * (2 * cov + c2)) / ((mu_p2 + mu_t2 + c1) * (var_p + var_t + c2))
    return s.flatten(1).mean(1)


class MetricTracker:
    """Running means, grouped by an arbitrary tag.

    Revisit queries and current-chunk queries are tracked apart, because the
    average over both hides the thing being measured: a model can hold its
    overall score steady while steadily losing the viewpoints it saw earliest.
    """

    def __init__(self):
        self.sums, self.counts = {}, {}

    def update(self, name: str, value, n: int = 1) -> None:
        if torch.is_tensor(value):
            value = float(value.mean())
        self.sums[name] = self.sums.get(name, 0.0) + value * n
        self.counts[name] = self.counts.get(name, 0) + n

    def mean(self, name: str) -> float:
        n = self.counts.get(name, 0)
        return self.sums[name] / n if n else float("nan")

    def as_dict(self) -> dict:
        return {k: self.mean(k) for k in self.sums}

    def reset(self) -> None:
        self.sums, self.counts = {}, {}
