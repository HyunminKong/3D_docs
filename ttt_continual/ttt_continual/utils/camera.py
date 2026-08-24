"""Camera geometry: rays, pose normalisation, intrinsic rescaling.

Poses arrive in whatever frame the dataset uses, which for driving data is a
per-location map frame with coordinates in the hundreds of metres. Everything
downstream assumes a scene of roughly unit size centred on the first camera, so
normalisation happens once per episode and is shared by every chunk in it --
recomputing it per chunk would silently change the scale a scene is
reconstructed at, and any quality measured across chunks would not be comparable.
"""

from typing import Tuple

import numpy as np
import torch


def compute_rays(fxfycxcy: torch.Tensor, c2w: torch.Tensor, h: int, w: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel ray origins and directions.

    Args:
        fxfycxcy: (b, v, 4) intrinsics for each view.
        c2w: (b, v, 4, 4) camera-to-world.
    Returns:
        ray_o, ray_d: (b, v, 3, h, w). Directions are normalised.
    """
    b, v = fxfycxcy.shape[0], fxfycxcy.shape[1]
    device = c2w.device

    idx_x = torch.arange(w, device=device)[None, :].expand(h, -1)
    idx_y = torch.arange(h, device=device)[:, None].expand(-1, w)
    idx_x = idx_x.reshape(1, -1).expand(b * v, -1).float()
    idx_y = idx_y.reshape(1, -1).expand(b * v, -1).float()

    intr = fxfycxcy.reshape(b * v, 4)
    pose = c2w.reshape(b * v, 4, 4)

    # pixel centres, hence the half-pixel offset
    x = (idx_x + 0.5 - intr[:, 2:3]) / intr[:, 0:1]
    y = (idx_y + 0.5 - intr[:, 3:4]) / intr[:, 1:2]
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], dim=1)                       # (b*v, 3, h*w)

    ray_d = torch.bmm(pose[:, :3, :3], dirs)                   # rotate into world
    ray_d = ray_d / (ray_d.norm(dim=1, keepdim=True) + 1e-8)
    ray_o = pose[:, :3, 3:4].expand_as(ray_d)

    ray_o = ray_o.reshape(b, v, 3, h, w)
    ray_d = ray_d.reshape(b, v, 3, h, w)
    return ray_o, ray_d


def plucker(ray_o: torch.Tensor, ray_d: torch.Tensor) -> torch.Tensor:
    """Plucker coordinates (d, o x d), which are invariant to sliding the origin
    along the ray and so encode the ray itself rather than an arbitrary point on
    it."""
    moment = torch.cross(ray_o, ray_d, dim=2)
    return torch.cat([ray_d, moment], dim=2)                   # (b, v, 6, h, w)


def normalise_poses(c2w: np.ndarray, method: str = "first_cam"
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Put an episode in a canonical frame of roughly unit extent.

    Returns the transformed poses, the inverse of the applied rotation-translation
    (so predictions can be pushed back to world coordinates), and the scale that
    was divided out.
    """
    c2w = np.asarray(c2w, dtype=np.float64).copy()
    if method == "first_cam":
        anchor = c2w[0].copy()
    elif method == "mean_cam":
        centre = c2w[:, :3, 3].mean(0)
        forward = c2w[:, :3, 2].sum(0)
        up = c2w[:, :3, 1].sum(0)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        true_up = np.cross(forward, right)
        anchor = np.eye(4)
        anchor[:3, :3] = np.stack([right, true_up, forward], axis=1)
        anchor[:3, 3] = centre
    else:
        raise ValueError(f"unknown normalisation: {method}")

    c2w = np.linalg.inv(anchor) @ c2w
    scale = float(np.abs(c2w[:, :3, 3]).max())
    scale = max(scale, 1e-6)
    c2w[:, :3, 3] /= scale
    return c2w, np.linalg.inv(anchor), 1.0 / scale


def rescale_intrinsics(intr: np.ndarray, src_hw: Tuple[int, int],
                       dst_hw: Tuple[int, int]) -> np.ndarray:
    """Adjust fx, fy, cx, cy for a resize. Axes scale independently, matching a
    plain resize rather than an aspect-preserving one."""
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    out = np.asarray(intr, dtype=np.float64).copy()
    out[..., 0] *= sx
    out[..., 1] *= sy
    out[..., 2] *= sx
    out[..., 3] *= sy
    return out
