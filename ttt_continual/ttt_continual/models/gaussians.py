"""Decoding tokens into Gaussians, and rendering them.

Positions are parameterised as a depth along the query pixel's own ray rather
than as a free point in space. It costs two degrees of freedom per primitive and
buys a great deal: the prediction cannot wander off the ray it was decoded from,
which is what keeps a feed-forward reconstruction stable when the same surface
is seen from a new angle.

For dynamic scenes each query carries the timestamp it is asking about, so the
decoded set describes the scene at that moment rather than an average over the
episode. Nothing here has to be told which parts move; the time embedding lets
the trunk represent them differently if that lowers the loss.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gsplat import rasterization
except ImportError:                                            # pragma: no cover
    rasterization = None


def sinusoidal_time(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Fourier features of a scalar time in [0, 1]."""
    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32)
                     * (-torch.log(torch.tensor(1000.0, device=t.device)) / max(half - 1, 1)))
    ang = t.unsqueeze(-1).float() * freq * torch.pi * 2.0
    out = torch.cat([ang.sin(), ang.cos()], dim=-1)
    if out.shape[-1] < dim:
        out = F.pad(out, (0, dim - out.shape[-1]))
    return out


class GaussianHead(nn.Module):
    """Token -> a small set of Gaussians on that token's rays."""

    def __init__(self, dim: int, patch_size: int, sh_degree: int = 1,
                 max_depth: float = 10.0, scale_bias: float = -4.0,
                 opacity_bias: float = -2.0, init_std: float = 1e-3):
        super().__init__()
        self.patch_size = patch_size
        self.sh_degree = sh_degree
        self.n_sh = (sh_degree + 1) ** 2
        self.max_depth = max_depth
        self.scale_bias = scale_bias
        self.opacity_bias = opacity_bias
        # depth(1) + sh(3 * n_sh) + scale(3) + rotation(4) + opacity(1)
        self.per_gaussian = 1 + 3 * self.n_sh + 3 + 4 + 1
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.per_gaussian * patch_size ** 2, bias=False)
        # Small, not zero. A zero last layer is a common way to start a residual
        # branch at identity, but here it makes the head's output independent of
        # its input: every token decodes to the same Gaussian, no change anywhere
        # upstream can alter a render, and the gradient reaching the trunk is
        # exactly zero. It also silently empties the reusability objective, whose
        # whole content is that different skills produce different renders.
        nn.init.normal_(self.proj.weight, std=init_std)

    def forward(self, tokens: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (b, n_tokens, dim) query tokens.
            ray_o, ray_d: (b, n_gauss, 3), one ray per emitted Gaussian, already
                flattened in the same order the head unpacks.
        """
        b = tokens.shape[0]
        raw = self.proj(self.norm(tokens)).reshape(b, -1, self.per_gaussian)
        assert raw.shape[1] == ray_o.shape[1], "one ray per Gaussian is required"

        idx = 0
        depth = raw[..., idx:idx + 1]; idx += 1
        sh = raw[..., idx:idx + 3 * self.n_sh]; idx += 3 * self.n_sh
        scale = raw[..., idx:idx + 3]; idx += 3
        rot = raw[..., idx:idx + 4]; idx += 4
        opacity = raw[..., idx:idx + 1]

        depth = torch.sigmoid(depth) * self.max_depth
        xyz = ray_o + depth * ray_d
        sh = sh.reshape(b, -1, self.n_sh, 3)
        # identity quaternion at initialisation, since proj starts at zero
        rot = rot + torch.tensor([1.0, 0.0, 0.0, 0.0], device=rot.device)
        return {
            "xyz": xyz,
            "sh": sh,
            "scale": scale + self.scale_bias,
            "rotation": rot,
            "opacity": opacity + self.opacity_bias,
            "depth": depth,
        }


def render(gaussians: Dict[str, torch.Tensor], c2w: torch.Tensor, intr: torch.Tensor,
           height: int, width: int, sh_degree: int = 1,
           near: float = 0.05, far: float = 200.0,
           background: float = 1.0) -> Dict[str, torch.Tensor]:
    """Rasterise one batch element's Gaussians into a set of views.

    Args:
        gaussians: as returned by GaussianHead, batch size 1 in the first dim.
        c2w: (v, 4, 4) cameras to render from.
        intr: (v, 4) fx, fy, cx, cy.
    Returns:
        rgb (v, 3, h, w) and depth (v, 1, h, w).
    """
    if rasterization is None:
        raise RuntimeError("gsplat is required for rendering")

    xyz = gaussians["xyz"][0].float()
    sh = gaussians["sh"][0].float()
    scale = gaussians["scale"][0].float().exp()
    rot = F.normalize(gaussians["rotation"][0].float(), dim=-1)
    opacity = gaussians["opacity"][0].float().sigmoid().squeeze(-1)

    w2c = torch.linalg.inv(c2w.float())
    k = torch.zeros(c2w.shape[0], 3, 3, device=c2w.device, dtype=torch.float32)
    k[:, 0, 0] = intr[:, 0]; k[:, 1, 1] = intr[:, 1]
    k[:, 0, 2] = intr[:, 2]; k[:, 1, 2] = intr[:, 3]; k[:, 2, 2] = 1.0

    out, _, _ = rasterization(
        xyz, rot, scale, opacity, sh, w2c, k, width, height,
        sh_degree=sh_degree, near_plane=near, far_plane=far,
        render_mode="RGB+D", eps2d=0.3, rasterize_mode="classic",
        backgrounds=torch.full((c2w.shape[0], 3), background, device=xyz.device),
    )
    rgb = out[..., :3].permute(0, 3, 1, 2)
    depth = out[..., 3:4].permute(0, 3, 1, 2)
    return {"rgb": rgb, "depth": depth}
