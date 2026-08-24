"""Convention self-test: cross-project GT depth between frame pairs.

A wrong pose handedness / optical-frame convention is the most common silent bug
in this kind of probe -- it produces plausible-looking training curves and
nonsense geometry. This script backprojects frame A's GT depth into world space,
projects it into frame B, and compares the projected range against B's own depth
map. Correct conventions give errors at the sensor-noise level (~cm); a wrong
convention gives metres.

Usage:
    python scripts/check_data.py --dataset nrgbd --scene staircase
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from dnpc.data.base import backproject  # noqa: E402
from dnpc.data.nrgbd import NRGBDSequence  # noqa: E402
from dnpc.data.tum import TUMSequence  # noqa: E402

NRGBD_ROOT = "/home/khm/3D_4D/FastVGGT/data/neural_rgbd_data"
TUM_ROOT = "/home/khm/3D_4D/data/tum"


def cross_project_error(seq, ia: int, ib: int):
    A, B = seq[ia], seq[ib]
    pts, _ = backproject(A.depth_gt, A.K, A.c2w, stride=4)
    if len(pts) < 100:
        return None
    w2c = np.linalg.inv(B.c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-3
    cam = cam[front]
    uv = cam[:, :2] / cam[:, 2:3] @ B.K[:2, :2].T + B.K[:2, 2]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inb = (u >= 0) & (u < seq.W) & (v >= 0) & (v < seq.H)
    if inb.sum() < 100:
        return None
    dB = B.depth_gt[v[inb], u[inb]]
    zA = cam[inb, 2]
    ok = dB > 0
    if ok.sum() < 100:
        return None
    # Occlusion makes the distribution one-sided; the median is the robust stat.
    return float(np.median(np.abs(zA[ok] - dB[ok]))), int(ok.sum()), float(inb.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nrgbd", "tum"], required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 5, 20])
    args = ap.parse_args()

    if args.dataset == "nrgbd":
        seq = NRGBDSequence(NRGBD_ROOT, args.scene, stride=args.stride)
    else:
        seq = TUMSequence(TUM_ROOT, args.scene, stride=args.stride)

    c = seq.cam_centers()
    span = float(np.linalg.norm(c.max(0) - c.min(0)))
    step = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)))
    print(f"{seq.name}: {len(seq)} frames  {seq.W}x{seq.H}  "
          f"traj_span={span:.2f}m  median_step={step*100:.2f}cm  scene_scale={seq.scene_scale():.2f}m")

    f0 = seq[0]
    d = f0.depth_gt[f0.depth_gt > 0]
    print(f"  depth p1/p50/p99 = {np.percentile(d,1):.2f}/{np.percentile(d,50):.2f}/{np.percentile(d,99):.2f} m")
    di = f0.depth_init[f0.depth_gt > 0]
    print(f"  init-vs-GT depth: median |dz| = {np.median(np.abs(di-d))*1000:.1f} mm")

    print("  cross-projection consistency (median |z_proj - z_B|):")
    worst = 0.0
    for g in args.gaps:
        ib = min(g, len(seq) - 1)
        r = cross_project_error(seq, 0, ib)
        if r is None:
            print(f"    gap {g:3d}: insufficient overlap")
            continue
        err, n, cov = r
        worst = max(worst, err)
        print(f"    gap {g:3d}: {err*1000:7.1f} mm   (n={n}, in-bounds {cov*100:.0f}%)")

    verdict = "PASS" if worst < 0.15 else "FAIL"
    print(f"  => {verdict} (threshold 150 mm; a wrong convention typically gives >1000 mm)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
