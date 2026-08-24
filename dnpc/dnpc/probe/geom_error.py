"""Derived per-Gaussian quantities: observation geometry and geometric error.

Two groups of numbers are produced here.

**Observation geometry**, from the visibility record ``vis[N, T]`` and the camera
centres. Exact pairwise diameters would be O(N T^2); instead each set diameter is
estimated by its maximum extent over a fixed bundle of directions, which is a
lower bound that is tight to a couple of percent for the smooth trajectories in
these sequences.

  ``B_acc``     diameter of the observing camera centres (the plan's raw baseline)
  ``B_perp``    the same, but projected onto the plane perpendicular to the
                first-observation ray. This is the component that actually
                triangulates: under forward motion ``B_acc`` can be large while
                ``B_perp`` ~ 0, and only ``B_perp`` enters sigma_z = z^2 sigma_d/(f B).
  ``alpha_max`` maximum angle subtended at the Gaussian by two observing cameras

**Geometric error**, against a voxel-downsampled GT point cloud with PCA normals.
The point-to-plane residual is decomposed along and across the first-observation
ray, which is what tests the anisotropy claim (Axis A) directly:

    err_radial  = |d * (n.v)|          err_lateral = |d| * sqrt(1 - (n.v)^2)

``n_dot_v`` is kept so that grazing surfaces -- where point-to-plane structurally
cannot see radial displacement -- can be filtered in analysis.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


# ------------------------------------------------------------------ geometry
def _dir_bundle(n: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(n, 3, generator=g)
    return (d / d.norm(dim=-1, keepdim=True)).to(device)


@torch.no_grad()
def observation_geometry(
    vis: torch.Tensor,  # [N, T] bool
    means: torch.Tensor,  # [N, 3]
    ray_first: torch.Tensor,  # [N, 3] unit
    cam_centers: torch.Tensor,  # [T, 3]
    chunk: int = 4096,
    n_dirs: int = 32,
    n_dirs_2d: int = 16,
):
    N, T = vis.shape
    dev = means.device
    dirs = _dir_bundle(n_dirs, dev)  # [D, 3]
    theta = torch.arange(n_dirs_2d, device=dev, dtype=torch.float32) * (np.pi / n_dirs_2d)
    cos_t, sin_t = theta.cos(), theta.sin()

    out = {k: torch.zeros(N, device=dev) for k in
           ("B_acc", "B_perp", "alpha_max", "z_mean", "n_obs", "first_obs", "last_obs")}
    NEG = torch.finfo(torch.float32).min

    proj_all = cam_centers @ dirs.T  # [T, D]
    tcol = torch.arange(T, device=dev, dtype=torch.float32)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        m = vis[s:e]  # [n, T]
        n = e - s
        cnt = m.sum(1)
        out["n_obs"][s:e] = cnt.float()
        any_obs = cnt > 0
        out["first_obs"][s:e] = torch.where(any_obs, (tcol + (1 - m.float()) * 1e9).min(1).values, torch.tensor(-1.0, device=dev))
        out["last_obs"][s:e] = torch.where(any_obs, (tcol - (1 - m.float()) * 1e9).max(1).values, torch.tensor(-1.0, device=dev))

        # B_acc: extent of observing camera centres over the direction bundle
        p = torch.where(m[:, :, None], proj_all[None], torch.full_like(proj_all[None], NEG))
        hi = p.max(1).values
        p = torch.where(m[:, :, None], proj_all[None], torch.full_like(proj_all[None], -NEG))
        lo = p.min(1).values
        out["B_acc"][s:e] = torch.where(cnt > 1, (hi - lo).max(1).values, torch.zeros(n, device=dev))

        # per-Gaussian quantities need the relative geometry
        rel = cam_centers[None] - means[s:e, None]  # [n, T, 3]
        dist = rel.norm(dim=-1).clamp_min(1e-8)
        out["z_mean"][s:e] = torch.where(
            any_obs, (dist * m).sum(1) / cnt.clamp_min(1).float(), torch.zeros(n, device=dev)
        )

        # alpha_max: chord diameter of the bearing set on the unit sphere
        bear = rel / dist[..., None]
        bp = torch.einsum("ntc,dc->ntd", bear, dirs)
        hi = torch.where(m[:, :, None], bp, torch.full_like(bp, NEG)).max(1).values
        lo = torch.where(m[:, :, None], bp, torch.full_like(bp, -NEG)).min(1).values
        chord = (hi - lo).max(1).values.clamp(0, 2)
        out["alpha_max"][s:e] = torch.where(
            cnt > 1, 2 * torch.asin((chord / 2).clamp(0, 1)), torch.zeros(n, device=dev)
        )

        # B_perp: 2D diameter in the plane perpendicular to the first ray
        v = ray_first[s:e]
        tmp = torch.zeros_like(v)
        tmp[:, 0] = 1.0
        alt = torch.zeros_like(v)
        alt[:, 1] = 1.0
        helper = torch.where((v[:, 0].abs() > 0.9)[:, None], alt, tmp)
        e1 = torch.cross(v, helper, dim=-1)
        e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        e2 = torch.cross(v, e1, dim=-1)
        c1 = torch.einsum("tc,nc->nt", cam_centers, e1)
        c2 = torch.einsum("tc,nc->nt", cam_centers, e2)
        pp = c1[:, :, None] * cos_t[None, None] + c2[:, :, None] * sin_t[None, None]
        hi = torch.where(m[:, :, None], pp, torch.full_like(pp, NEG)).max(1).values
        lo = torch.where(m[:, :, None], pp, torch.full_like(pp, -NEG)).min(1).values
        out["B_perp"][s:e] = torch.where(cnt > 1, (hi - lo).max(1).values, torch.zeros(n, device=dev))
    return {k: v.cpu().numpy() for k, v in out.items()}


# --------------------------------------------------------------- GT geometry
def build_gt_cloud(seq, frame_stride: int = 5, pixel_stride: int = 3, voxel: float = 0.005):
    """Fuse GT depth into a voxel-downsampled world point cloud with PCA normals."""
    from ..data.base import backproject

    pts = []
    for i in range(0, len(seq), frame_stride):
        f = seq[i]
        p, _ = backproject(f.depth_gt, f.K, f.c2w, stride=pixel_stride)
        pts.append(p)
    P = np.concatenate(pts, 0)

    q = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    P = np.ascontiguousarray(P[np.sort(idx)])

    tree = cKDTree(P)
    _, nn = tree.query(P, k=16, workers=-1)
    nb = P[nn]  # [M, 16, 3]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb) / nb.shape[1]
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0].astype(np.float32)  # smallest-eigenvalue direction
    return P.astype(np.float32), normals, tree


def geometric_error(means: np.ndarray, ray_first: np.ndarray, P, normals, tree,
                    max_dist: float = 0.5):
    """Point-to-plane residual to the GT surface, split into radial / lateral."""
    d_nn, nn = tree.query(means, k=1, workers=-1)
    n = normals[nn]
    signed = np.einsum("ij,ij->i", means - P[nn], n)
    n_dot_v = np.einsum("ij,ij->i", n, ray_first)
    a = np.abs(signed)
    lat_frac = np.sqrt(np.clip(1.0 - n_dot_v**2, 0.0, 1.0))
    return {
        "err_p2pl": a.astype(np.float32),
        "err_radial": (a * np.abs(n_dot_v)).astype(np.float32),
        "err_lateral": (a * lat_frac).astype(np.float32),
        "err_nn": d_nn.astype(np.float32),
        "n_dot_v": n_dot_v.astype(np.float32),
        "gt_valid": (d_nn < max_dist),
    }
