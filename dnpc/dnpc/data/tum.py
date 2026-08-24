"""TUM RGB-D loader (rgb/depth/groundtruth association by timestamp).

TUM ships raw Kinect depth, so unlike NRGBD there is no clean/noisy pair. To keep
the Stage 0 setup comparable across datasets we synthesise the *initialisation*
depth by perturbing the sensor depth with a stereo-consistent noise model::

    sigma_z(z) = disp_noise_coeff * z^2          [metres]

which is the ``sigma_z = z^2 sigma_d / (f B)`` law with the Kinect's baseline and
focal folded into a single coefficient (~2.3e-3 m^-1 for f=580px, B=7.5cm,
sigma_d=0.1px). Gaussians spawn from the perturbed depth and are scored against
the sensor depth. This coefficient is an explicit experimental knob -- it is the
``sigma_d`` that appears in the theory, so sweeping it is a direct test of the
predicted scaling rather than a nuisance parameter.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .base import Frame, Sequence

DEPTH_SCALE = 5000.0
MAX_DEPTH_M = 10.0

# Per-Freiburg-camera intrinsics published with the dataset.
INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _read_list(path: str):
    out = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split()
            out.append((float(parts[0]), parts[1:]))
    return out


def _associate(a, b, max_dt: float):
    """Greedy nearest-timestamp matching of two sorted (ts, payload) lists."""
    tb = np.array([t for t, _ in b])
    pairs = []
    for ta, pa in a:
        j = int(np.argmin(np.abs(tb - ta)))
        if abs(tb[j] - ta) <= max_dt:
            pairs.append((ta, pa, b[j][1]))
    return pairs


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        np.float32,
    )


class TUMSequence(Sequence):
    def __init__(
        self,
        root: str,
        seq: str,
        stride: int = 1,
        max_frames: int | None = None,
        max_dt: float = 0.02,
        disp_noise_coeff: float = 2.3e-3,
        seed: int = 0,
    ):
        self.dir = os.path.join(root, seq)
        self.name = f"tum/{seq}"
        self.disp_noise_coeff = disp_noise_coeff
        self._rng_seed = seed

        rgb = _read_list(os.path.join(self.dir, "rgb.txt"))
        dep = _read_list(os.path.join(self.dir, "depth.txt"))
        gt = _read_list(os.path.join(self.dir, "groundtruth.txt"))

        rd = _associate(rgb, dep, max_dt)  # (ts, [rgb_path], [depth_path])
        rd_as_list = [(ts, [rp[0], dp[0]]) for ts, rp, dp in rd]
        full = _associate(rd_as_list, gt, max_dt)  # (ts, [rgb, depth], [tx..qw])

        full = full[::stride]
        if max_frames is not None:
            full = full[:max_frames]

        self.records = []
        poses = []
        for k, (ts, paths, pose) in enumerate(full):
            tx, ty, tz, qx, qy, qz, qw = [float(v) for v in pose]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
            T[:3, 3] = (tx, ty, tz)
            poses.append(T)
            self.records.append((k, paths[0], paths[1]))
        if not poses:
            raise RuntimeError(f"no associated frames for {seq}")
        self.poses = np.stack(poses)

        fam = next((k for k in INTRINSICS if k in seq), None)
        if fam is None:
            raise ValueError(f"cannot infer Freiburg camera family from '{seq}'")
        fx, fy, cx, cy = INTRINSICS[fam]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        self.H, self.W = 480, 640

    def __len__(self):
        return len(self.records)

    def cam_centers(self):
        return self.poses[:, :3, 3].copy()

    def __getitem__(self, i: int) -> Frame:
        _, rgb_rel, dep_rel = self.records[i]
        rgb = cv2.imread(os.path.join(self.dir, rgb_rel), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d = cv2.imread(os.path.join(self.dir, dep_rel), cv2.IMREAD_UNCHANGED)
        d = d.astype(np.float32) / DEPTH_SCALE
        d[(d > MAX_DEPTH_M) | (d < 1e-3)] = 0.0

        # Deterministic per-frame perturbation so reruns are reproducible.
        rng = np.random.default_rng(self._rng_seed * 100003 + i)
        noise = rng.standard_normal(d.shape).astype(np.float32) * (self.disp_noise_coeff * d**2)
        d_init = np.where(d > 0, np.clip(d + noise, 1e-3, MAX_DEPTH_M), 0.0).astype(np.float32)

        return Frame(
            idx=i, src_idx=i, rgb=rgb, depth_gt=d, depth_init=d_init,
            c2w=self.poses[i], K=self.K,
        )
