"""Offline-only frozen VGGT geometry pseudo-label provider.

This module is deliberately separate from ``FrozenVGGTFeatures``.  Its heads
are used only to bootstrap a non-degenerate custom decoder for controlled
pre-framework experiments; they are never an online TTT signal or a fallback
prediction head at deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenVGGTDepthTeacher(nn.Module):
    """Return low-resolution depth pseudo-labels from a frozen full VGGT."""

    def __init__(self, checkpoint: str | Path, *, repo_root: str | Path = "FastVGGT"):
        super().__init__()
        repo_root = Path(repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vggt.models.vggt import VGGT

        self.model = VGGT(enable_camera=True, enable_point=True, enable_depth=True,
                          enable_track=False, merging=24)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor, output_size: tuple[int, int]) -> dict[str, Tensor]:
        if not images.is_cuda:
            raise RuntimeError("The frozen VGGT pseudo-label teacher is intended for CUDA bootstrap only")
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("VGGT inputs must be divisible by 14")
        self.model.update_patch_dimensions(width // 14, height // 14)
        # Official FastVGGT inference keeps parameters in fp32 and uses bf16
        # autocast.  Casting the whole model breaks its DPT heads.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model(images.bfloat16())
        depth = prediction["depth"][..., 0].float()
        batch, views = depth.shape[:2]
        depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1), size=output_size, mode="area")
        depth = depth.reshape(batch, views, *output_size)
        confidence = prediction["depth_conf"].float()
        confidence = F.interpolate(confidence.flatten(0, 1).unsqueeze(1), size=output_size, mode="area")
        confidence = confidence.reshape(batch, views, *output_size)
        # VGGT's camera head predicts a 3x4 w2c transform in its own metric
        # gauge.  Retaining that gauge with its depth is essential: mixing a
        # foundation depth prior with dataset-scale poses was the source of the
        # zero-valid-pixel degeneracy found by the objective-health audit.
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        w2c_3x4, intrinsics_matrix = pose_encoding_to_extri_intri(
            prediction["pose_enc"], image_size_hw=(height, width)
        )
        w2c = F.pad(w2c_3x4.float(), (0, 0, 0, 1))
        w2c[..., 3, 3] = 1
        intrinsics = torch.stack((
            intrinsics_matrix[..., 0, 0], intrinsics_matrix[..., 1, 1],
            intrinsics_matrix[..., 0, 2], intrinsics_matrix[..., 1, 2],
        ), dim=-1).float()
        return {"depth": depth, "confidence": confidence, "w2c": w2c, "intrinsics": intrinsics}


class FrozenVGGTGeometryTracker(nn.Module):
    """Frozen camera-and-track prior for controlled geometry-only TTT probes.

    Unlike ``FrozenVGGTDepthTeacher``, this module supplies no depth target.
    Its fixed correspondences and camera gauge let a new head's *own* depth
    prediction be tested with 3D track consistency.  It is an experimental
    foundation prior, not a fallback decoder in the proposed architecture.
    """

    def __init__(self, checkpoint: str | Path, *, repo_root: str | Path = "FastVGGT"):
        super().__init__()
        repo_root = Path(repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from vggt.models.vggt import VGGT

        self.model = VGGT(enable_camera=True, enable_point=False, enable_depth=False,
                          enable_track=True, merging=24)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights, strict=False)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor, query_points: Tensor) -> dict[str, Tensor]:
        if not images.is_cuda:
            raise RuntimeError("The frozen VGGT geometry tracker is intended for CUDA probes only")
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError("VGGT inputs must be divisible by 14")
        self.model.update_patch_dimensions(width // 14, height // 14)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self.model(images.bfloat16(), query_points=query_points.float())
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        w2c_3x4, intrinsics_matrix = pose_encoding_to_extri_intri(
            prediction["pose_enc"], image_size_hw=(height, width)
        )
        w2c = F.pad(w2c_3x4.float(), (0, 0, 0, 1))
        w2c[..., 3, 3] = 1
        intrinsics = torch.stack((
            intrinsics_matrix[..., 0, 0], intrinsics_matrix[..., 1, 1],
            intrinsics_matrix[..., 0, 2], intrinsics_matrix[..., 1, 2],
        ), dim=-1).float()
        return {"w2c": w2c, "intrinsics": intrinsics, "track": prediction["track"].float(),
                "visibility": prediction["vis"].float(), "confidence": prediction["conf"].float()}
