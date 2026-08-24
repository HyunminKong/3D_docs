"""Cross-scene replication of the headline statistic.

For each scene, the direct matched-age contrast between the stride-2 and stride-8
runs: same trajectory, same age in stream frames, same optimiser steps per
Gaussian, but 2-4x the accumulated baseline. The implied exponent

    ln(err_8 / err_2) / ln(B_8 / B_2)

is 0 if parallax does nothing and -1 if error is triangulation-limited. CIs are
bootstrapped over Gaussians within each age bin.

    python scripts/stage0_summary.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"
AGE_BINS = [(5, 15), (15, 30), (30, 60), (60, 120)]


def scene_prefixes():
    seen = set()
    for p in sorted(glob.glob(f"{OUT}/logs/*_str8_checkpoints.parquet")):
        seen.add(os.path.basename(p).replace("_str8_checkpoints.parquet", ""))
    return sorted(seen)


def load_pair(prefix):
    def rd(tag):
        p = f"{OUT}/logs/{prefix}_{tag}_checkpoints.parquet"
        return pd.read_parquet(p) if os.path.exists(p) else None

    lo = rd("str2") if rd("str2") is not None else rd("main")
    hi = rd("str8")
    if lo is None or hi is None:
        return None
    out = []
    for d, st in ((lo, 2), (hi, 8)):
        d = d[d.gt_valid & (d.B_perp > 1e-4) & (d.err_radial > 0) & (d.n_obs >= 2)].copy()
        d["stride"] = st
        d["age"] = d.frame - d.birth_frame
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    return df[df.contrib > np.percentile(df.contrib, 20)]


def exponents(df, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in AGE_BINS:
        s = df[(df.age > lo) & (df.age <= hi)]
        a, b = s[s.stride == 2], s[s.stride == 8]
        if len(a) < 200 or len(b) < 200:
            continue
        B2, B8 = a.B_perp.median(), b.B_perp.median()
        if not (B8 / B2 > 1.2):  # need a real baseline contrast
            continue
        e2, e8 = a.err_radial.median(), b.err_radial.median()
        p = np.log(e8 / e2) / np.log(B8 / B2)
        bs = [np.log(np.median(rng.choice(b.err_radial.values, len(b)))
                     / np.median(rng.choice(a.err_radial.values, len(a)))) / np.log(B8 / B2)
              for _ in range(n_boot)]
        ci = np.percentile(bs, [2.5, 97.5])
        rows.append({"age": f"({lo},{hi}]", "n2": len(a), "n8": len(b),
                     "B2": B2, "B8": B8, "B_ratio": B8 / B2,
                     "err2_mm": e2 * 1e3, "err8_mm": e8 * 1e3,
                     "exponent": p, "lo": ci[0], "hi": ci[1]})
    return pd.DataFrame(rows)


def main():
    all_rows, summary = [], []
    for pref in scene_prefixes():
        df = load_pair(pref)
        if df is None:
            continue
        tab = exponents(df)
        if tab.empty:
            print(f"[{pref}] no age bin with a usable baseline contrast")
            continue
        tab.insert(0, "scene", pref)
        all_rows.append(tab)
        print(f"\n### {pref}")
        print(tab.drop(columns="scene").to_string(index=False, float_format="%.3f"))
        summary.append({"scene": pref, "mean_exponent": float(tab.exponent.mean()),
                        "min": float(tab.exponent.min()), "max": float(tab.exponent.max()),
                        "n_bins": int(len(tab))})

    if not all_rows:
        print("no scenes ready")
        return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(f"{OUT}/tables/stride_exponents_all_scenes.csv", index=False)
    s = pd.DataFrame(summary)
    print("\n=== replication across scenes (theory: -1.0, no effect: 0.0) ===")
    print(s.to_string(index=False, float_format="%.3f"))
    grand = float(full.exponent.mean())
    print(f"\n  grand mean exponent = {grand:+.3f}  over {len(full)} scene x age cells")
    with open(f"{OUT}/tables/verdict_replication.json", "w") as f:
        json.dump({"per_scene": summary, "grand_mean_exponent": grand,
                   "theory": -1.0, "n_cells": int(len(full)),
                   "H1_supported": bool(full.exponent.mean() < -0.5)}, f, indent=2)


if __name__ == "__main__":
    main()
