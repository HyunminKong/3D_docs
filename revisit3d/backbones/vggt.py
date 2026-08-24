"""Frozen VGGT token extractor used by Revisit3D.

The wrapper deliberately exposes intermediate geometry tokens rather than
VGGT's pretrained prediction heads.  Revisit3D's geometry head is therefore
the sole place where compact test-time state is injected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn


class FrozenVGGTFeatures(nn.Module):
    """Extract last-layer patch tokens ``[B, V, P, 2048]`` without gradients."""

    feature_dim = 2048

    def __init__(self, checkpoint: str | Path, *, repo_root: str | Path = "FastVGGT"):
        super().__init__()
        repo_root = Path(repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"FastVGGT checkout not found: {repo_root}")
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vggt.models.vggt import VGGT

        # Heads are disabled: their pretrained task-specific decoding must not
        # become a hidden alternate adaptation route.
        self.model = VGGT(enable_camera=False, enable_point=False, enable_depth=False,
                          enable_track=False, merging=24)
        checkpoint = Path(checkpoint)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=False)
        # FastVGGT's aggregator explicitly casts images to bf16 internally, so
        # keeping the frozen convolution weights in fp32 would fail before any
        # feature reaches our head.
        self.model.eval().to(torch.bfloat16)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Return frozen patch features for ``images`` shaped [B,V,3,H,W]."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must be [batch, views, 3, height, width]")
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("VGGT input height and width must be divisible by 14")
        self.model.update_patch_dimensions(width // 14, height // 14)
        outputs, patch_start = self.model.aggregator(images)
        if not outputs:
            raise RuntimeError("VGGT aggregator returned no intermediate tokens")
        tokens = outputs[-1]
        return tokens[:, :, patch_start:].float()
