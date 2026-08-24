"""Neural RGB-D Surface Reconstruction (NRGBD) loader.

Layout per scene::

    images/img{i}.png
    depth/depth{i}.png              # clean GT, uint16 millimetres
    depth_with_noise/depth{i}.png   # realistic sensor noise -- used to *spawn*
    poses.txt                       # 4x4 c2w per frame, OpenGL convention, may contain nan

Conventions follow FastVGGT's verified loader (``FastVGGT/eval/data.py``):
intrinsics are fixed at fx=fy=554.2562584220408, cx=320, cy=240; depth is
millimetres; poses need the OpenGL->OpenCV flip ``pose[:, 1:3] *= -1``.

The clean/noisy split matters for Stage 0: Gaussians are *initialised* from the
noisy depth so there is real error for multi-view optimisation to resolve, and
scored against the clean depth. Initialising from clean GT would make
``err_final`` measure optimiser drift rather than information sufficiency.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .base import Frame, Sequence

FX = FY = 554.2562584220408
CX, CY = 320.0, 240.0
MAX_DEPTH_M = 10.0


def _load_poses(path: str):
    with open(path) as f:
        lines = f.readlines()
    poses, valid = [], []
    for i in range(0, len(lines), 4):
        block = lines[i : i + 4]
        if len(block) < 4 or any("nan" in ln.lower() for ln in block):
            poses.append(np.eye(4, dtype=np.float32))
            valid.append(False)
            continue
        poses.append(np.array([[float(x) for x in ln.split()] for ln in block], np.float32))
        valid.append(True)
    return np.stack(poses), np.array(valid, bool)


class NRGBDSequence(Sequence):
    def __init__(
        self,
        root: str,
        scene: str,
        stride: int = 1,
        max_frames: int | None = None,
        init_depth: str = "depth_with_noise",
    ):
        self.root, self.scene = root, scene
        self.dir = os.path.join(root, scene)
        self.name = f"nrgbd/{scene}"
        self.init_depth_dir = init_depth

        n_img = len([f for f in os.listdir(os.path.join(self.dir, "images")) if f.endswith(".png")])
        poses, valid = _load_poses(os.path.join(self.dir, "poses.txt"))
        n = min(n_img, len(poses))

        idxs = [i for i in range(0, n, stride) if valid[i]]
        if max_frames is not None:
            idxs = idxs[:max_frames]
        self.idxs = idxs

        # OpenGL -> OpenCV: flip the y and z basis vectors of the camera frame.
        self.poses = poses[idxs].copy()
        self.poses[:, :, 1:3] *= -1.0

        self.K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float32)
        self.H, self.W = 480, 640

    def __len__(self):
        return len(self.idxs)

    def cam_centers(self):
        return self.poses[:, :3, 3].copy()

    def _read_depth(self, sub: str, src: int):
        p = os.path.join(self.dir, sub, f"depth{src}.png")
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(p)
        d = np.nan_to_num(d.astype(np.float32), nan=0.0) / 1000.0
        d[(d > MAX_DEPTH_M) | (d < 1e-3)] = 0.0
        return d

    def __getitem__(self, i: int) -> Frame:
        src = self.idxs[i]
        rgb = cv2.imread(os.path.join(self.dir, "images", f"img{src}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d_gt = self._read_depth("depth", src)
        d_init = self._read_depth(self.init_depth_dir, src) if self.init_depth_dir else d_gt.copy()
        if rgb.shape[:2] != d_gt.shape:
            rgb = cv2.resize(rgb, (d_gt.shape[1], d_gt.shape[0]))
        return Frame(
            idx=i, src_idx=src, rgb=rgb, depth_gt=d_gt, depth_init=d_init,
            c2w=self.poses[i], K=self.K,
        )
