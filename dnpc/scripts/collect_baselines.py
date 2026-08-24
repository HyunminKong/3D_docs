"""Aggregate per-run metrics into the age-freezing baseline table (plan 6.9).

The age baseline is the bar Stage 1 has to clear (H4). It is measured here on the
same streaming protocol and the same instrumentation path as the main run, so the
Stage 1 comparison is apples-to-apples.

Reported per plan 7: wall-clock ms/frame (never FLOPs), peak GPU memory, active
Gaussian fraction, Gaussian count, and quality.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"


def main():
    rows = []
    for p in sorted(glob.glob(f"{OUT}/tables/*_metrics.json")):
        with open(p) as f:
            d = json.load(f)
        m, cfg = d["metrics"], d["config"]
        if "smoke" in m["run"] or "noinstr" in m["run"]:
            continue
        rows.append({
            "run": m["run"], "scene": m["scene"], "tag": m["tag"],
            "freeze_age_k": m["freeze_age_k"],
            "psnr": m["psnr"], "depth_abs_rel": m["depth_abs_rel"],
            "n_gaussians": m["n_gaussians"],
            "ms_per_frame_median": m["ms_per_frame_median"],
            "peak_gpu_gb": m["peak_gpu_gb"],
            "train_wall_min": m["train_wall_s"] / 60.0,
            "window": cfg["window"], "replay": cfg["replay"],
            "iters_per_frame": cfg["iters_per_frame"],
        })
    if not rows:
        print("no runs yet")
        return
    df = pd.DataFrame(rows).sort_values(["scene", "freeze_age_k"])

    # Speedup / quality delta relative to the no-freezing run of the same scene.
    base = df[df.freeze_age_k == 0].set_index("scene")
    for c, newc in (("ms_per_frame_median", "speedup"), ("psnr", "d_psnr"),
                    ("depth_abs_rel", "d_absrel")):
        ref = df.scene.map(base[c])
        df[newc] = (ref / df[c]) if newc == "speedup" else (df[c] - ref)

    path = f"{OUT}/tables/age_baseline.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False, float_format="%.4g"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
