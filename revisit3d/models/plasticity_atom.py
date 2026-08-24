"""Spatially addressable fast state for EXP-006 test-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PlasticityAtom:
    xyz: Tensor
    scale: Tensor
    key: Tensor
    code: Tensor
    confidence: Tensor

    def detach(self) -> "PlasticityAtom":
        return PlasticityAtom(*(value.detach() for value in (
            self.xyz, self.scale, self.key, self.code, self.confidence,
        )))


class SpatialPlasticityHead(nn.Module):
    """Decode an 8-D per-token fast code into a bounded log-depth residual."""

    def __init__(
        self, feature_dim: int = 2048, key_dim: int = 64, code_dim: int = 8, hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.code_dim = code_dim
        self.feature_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.register_buffer("key_mean", torch.zeros(feature_dim), persistent=True)
        self.key_projection = nn.Linear(feature_dim, key_dim, bias=False)
        self.decoder_query = nn.Linear(feature_dim, key_dim)
        self.decoder = nn.Sequential(
            nn.Linear(key_dim + code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initialize_key_projection(self, components: Tensor, mean: Tensor | None = None) -> None:
        if components.shape != self.key_projection.weight.shape:
            raise ValueError(
                f"PCA components must be {tuple(self.key_projection.weight.shape)}, got {tuple(components.shape)}"
            )
        with torch.no_grad():
            self.key_projection.weight.copy_(components.to(self.key_projection.weight))
            if mean is not None:
                if mean.shape != self.key_mean.shape:
                    raise ValueError(f"PCA mean must be {tuple(self.key_mean.shape)}, got {tuple(mean.shape)}")
                self.key_mean.copy_(mean.to(self.key_mean))

    def appearance_key(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must be [batch, views, tokens, feature_dim]")
        normalized = self.feature_norm(features)
        return F.normalize(self.key_projection(normalized - self.key_mean), dim=-1)

    def initial_code(self, features: Tensor) -> Tensor:
        return torch.zeros(*features.shape[:3], self.code_dim, device=features.device, dtype=features.dtype)

    def log_depth_residual(self, features: Tensor, code: Tensor) -> Tensor:
        expected = (*features.shape[:3], self.code_dim)
        if code.shape != expected:
            raise ValueError(f"code must be {expected}, got {tuple(code.shape)}")
        query = self.decoder_query(self.feature_norm(features))
        zeros = torch.zeros_like(code)
        active = self.decoder(torch.cat((query, code), dim=-1))
        identity = self.decoder(torch.cat((query, zeros), dim=-1))
        return 0.5 * torch.tanh(active - identity)

    def depth(self, features: Tensor, base_depth: Tensor, code: Tensor) -> Tensor:
        residual = self.log_depth_residual(features, code)
        if base_depth.shape == residual.shape:
            base = base_depth
        elif base_depth.ndim == 4 and base_depth.shape[:2] == residual.shape[:2]:
            base = base_depth.flatten(2).unsqueeze(-1)
        else:
            raise ValueError("base_depth must be [B,V,P,1] or [B,V,H,W]")
        return base * residual.exp()

    def online_update(
        self,
        features: Tensor,
        base_depth: Tensor,
        initial_code: Tensor,
        objective: Callable[[Tensor, Tensor], Tensor],
        *,
        step_size: float = 0.05,
        steps: int = 1,
        retain_initial_gradient: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """First-order TTT: detach online gradients, retain only the code path."""
        code = initial_code if retain_initial_gradient else initial_code.detach()
        history = []
        for _ in range(steps):
            code = code.requires_grad_(True)
            depth = self.depth(features, base_depth, code)
            loss = objective(depth, code)
            if loss.ndim:
                raise ValueError("online objective must be scalar")
            gradient, = torch.autograd.grad(loss, code, create_graph=False, only_inputs=True)
            normalizer = gradient.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            code = (code - step_size * gradient.detach() / normalizer).clamp(-1, 1)
            if not retain_initial_gradient:
                code = code.detach()
            history.append(loss.detach())
        return code, history


def token_track_support(
    tracks: Tensor,
    visibility: Tensor,
    confidence: Tensor,
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> Tensor:
    """Nearest-token splat of frozen track evidence, returned as [B,V,P,1]."""
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks must be [B,V,N,2]")
    if visibility.shape != tracks.shape[:3] or confidence.shape != tracks.shape[:3]:
        raise ValueError("track evidence shapes do not match")
    image_h, image_w = image_size
    grid_h, grid_w = grid_size
    x = ((tracks[..., 0] / max(image_w, 1)) * grid_w).floor().long().clamp(0, grid_w - 1)
    y = ((tracks[..., 1] / max(image_h, 1)) * grid_h).floor().long().clamp(0, grid_h - 1)
    index = y * grid_w + x
    valid = ((tracks[..., 0] >= 0) & (tracks[..., 0] <= image_w - 1)
             & (tracks[..., 1] >= 0) & (tracks[..., 1] <= image_h - 1))
    weight = visibility.to(tracks.dtype) * confidence.to(tracks.dtype) * valid.to(tracks.dtype)
    support = tracks.new_zeros(*tracks.shape[:2], grid_h * grid_w)
    count = tracks.new_zeros(*tracks.shape[:2], grid_h * grid_w)
    support.scatter_add_(-1, index, weight)
    count.scatter_add_(-1, index, valid.to(tracks.dtype))
    support = support / count.clamp_min(1)
    return support.unsqueeze(-1).clamp(0, 1)


def combine_atom_confidence(base_confidence: Tensor, track_support: Tensor) -> Tensor:
    if base_confidence.shape != track_support.shape:
        raise ValueError("base confidence and track support must have the same shape")
    return torch.sqrt(
        base_confidence.detach().clamp(1e-4, 1) * track_support.detach().clamp(1e-4, 1)
    )


def build_plasticity_atom(
    head: SpatialPlasticityHead,
    features: Tensor,
    xyz: Tensor,
    scale: Tensor,
    base_confidence: Tensor,
    tracks: Tensor,
    visibility: Tensor,
    track_confidence: Tensor,
    *,
    image_size: tuple[int, int],
    code: Tensor | None = None,
) -> PlasticityAtom:
    points = features.shape[2]
    side = int(points ** 0.5)
    if side * side != points:
        raise ValueError("atom construction requires a square token grid")
    support = token_track_support(
        tracks, visibility, track_confidence, image_size=image_size, grid_size=(side, side),
    )
    if code is None:
        code = head.initial_code(features)
    return PlasticityAtom(
        xyz=xyz.detach(),
        scale=scale.detach(),
        key=head.appearance_key(features),
        code=code,
        confidence=combine_atom_confidence(base_confidence, support),
    )
