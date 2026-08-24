"""Common frame/sequence interface for Stage 0 probe datasets.

Convention (enforced by every loader in this package):
  * ``c2w`` is camera-to-world, **OpenCV optical frame** (x right, y down, z forward).
  * ``K`` is a 3x3 pinhole intrinsic matrix for the stored image resolution.
  * depth is in **metres**; 0 marks invalid/missing.

``scripts/check_data.py`` cross-projects consecutive frames to verify the
convention actually holds for each dataset -- a wrong handedness is the single
most common silent bug in this kind of probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    idx: int  # index within the (already filtered) sequence
    src_idx: int  # index in the raw on-disk sequence, for traceability
    rgb: np.ndarray  # [H, W, 3] float32 in [0, 1]
    depth_gt: np.ndarray  # [H, W] float32 metres, 0 = invalid
    depth_init: np.ndarray  # [H, W] float32 metres, used only to spawn Gaussians
    c2w: np.ndarray  # [4, 4] float32, OpenCV convention
    K: np.ndarray  # [3, 3] float32

    @property
    def cam_center(self) -> np.ndarray:
        return self.c2w[:3, 3]


class Sequence:
    """Ordered, randomly-addressable stream of :class:`Frame`."""

    name: str
    H: int
    W: int
    K: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __getitem__(self, i: int) -> Frame:  # pragma: no cover - interface
        raise NotImplementedError

    def cam_centers(self) -> np.ndarray:
        """[T, 3] camera centres for every frame (cheap; poses are preloaded)."""
        raise NotImplementedError

    def scene_scale(self) -> float:
        c = self.cam_centers()
        return float(np.linalg.norm(c - c.mean(0), axis=1).mean())


def backproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, stride: int = 1):
    """Depth map -> world-space points. Returns (pts [M,3], pix_uv [M,2] int32)."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[vs, us]
    m = z > 0
    us, vs, z = us[m], vs[m], z[m]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], -1)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), np.stack([us, vs], -1).astype(np.int32)
