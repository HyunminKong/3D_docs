"""Per-Gaussian attribution of image-space quantities, without CUDA changes.

In alpha compositing the pixel value is linear in the per-Gaussian channel value::

    C_p = sum_g  c_g * w_{g,p},        w_{g,p} = a_{g,p} * prod_{j<g} (1 - a_{j,p})

and crucially the blending weight ``w`` does **not** depend on ``c``. So if we
rasterise a probe channel held at 1.0 as a leaf tensor and backpropagate a loss
that is a fixed weight map ``r`` dotted with the rendered image::

    L = sum_p r_p * C_p     =>     dL/dc_g = sum_p r_p * w_{g,p}

we get, exactly, "the ``r``-weighted sum over the pixels this Gaussian is
responsible for". Stacking several weight maps as separate channels yields all of
them from a *single* forward+backward:

    r = 1                    -> accumulated blending weight  ("contrib")
    r = |I_render - I_gt|    -> the Gaussian's share of photometric error
    r = |D_render - D_gt|    -> the Gaussian's share of depth error (along-ray)

This replaces what would otherwise be a rasteriser kernel modification, and it is
geometry-detached so it never perturbs the training gradients.
"""

from __future__ import annotations

import torch
from gsplat import rasterization


@torch.enable_grad()
def attribute(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,  # [4, 4] world-to-camera
    K: torch.Tensor,  # [3, 3]
    width: int,
    height: int,
    weight_maps: torch.Tensor,  # [C, H, W], detached
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> torch.Tensor:
    """Returns [N, C]: for each Gaussian and each weight map, sum_p r_p w_{g,p}."""
    n, c = means.shape[0], weight_maps.shape[0]
    probe = torch.ones(n, c, device=means.device, dtype=torch.float32, requires_grad=True)
    render, _, _ = rasterization(
        means=means.detach(),
        quats=quats.detach(),
        scales=scales.detach(),
        opacities=opacities.detach(),
        colors=probe,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=None,
        packed=True,
        near_plane=near_plane,
        far_plane=far_plane,
    )  # [1, H, W, C]
    loss = (render[0].permute(2, 0, 1) * weight_maps).sum()
    (grad,) = torch.autograd.grad(loss, probe)
    return grad.detach()


def error_weight_maps(
    rgb_render: torch.Tensor,  # [H, W, 3]
    rgb_gt: torch.Tensor,  # [H, W, 3]
    depth_render: torch.Tensor,  # [H, W]
    depth_gt: torch.Tensor,  # [H, W]
) -> torch.Tensor:
    """Stack the four probe channels used by the Stage 0 final evaluation pass.

    Channel order: [ones, photometric |dI|, depth |dz| (masked), valid mask].
    The depth channel needs its own normaliser because it is only defined where
    the GT depth is valid; dividing it by the all-pixel weight (channel 0) would
    bias it low wherever the sensor has holes.

        contrib  = ch0
        e_render = ch1 / ch0
        e_depth  = ch2 / ch3
    """
    valid = (depth_gt > 0).float()
    return torch.stack(
        [
            torch.ones_like(depth_gt),
            (rgb_render - rgb_gt).abs().mean(-1),
            (depth_render - depth_gt).abs() * valid,
            valid,
        ]
    ).detach()
