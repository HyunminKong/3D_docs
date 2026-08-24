"""Identification via frame stride: separate parallax from staleness.

Within a lineage, accumulated baseline is a near-monotone function of age, so the
fixed-effects coefficient on log B conflates "more parallax" with "older, staler,
no longer being optimised". Nothing in a single run can separate them.

Frame stride breaks the collinearity by construction. Streaming the same physical
trajectory at stride 2, 4 and 8 gives 1x, 2x and 4x the accumulated baseline at
the *same* age in stream frames and the same number of optimiser steps per
Gaussian. If parallax is what reduces error, error at matched age must fall with
stride. If it is really just age, matched-age error is stride-invariant.

    python scripts/stage0_instrument.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"
AGE_BINS = [(5, 15), (15, 30), (30, 60), (60, 120)]


def resolve_runs(prefix):
    """Prefer the matched str2/4/8 triple; fall back to the older `main` tag for
    stride 2 where a dedicated str2 run was not made (staircase)."""
    out = []
    for st in (2, 4, 8):
        for tag in ((f"str{st}", "main") if st == 2 else (f"str{st}",)):
            p = f"{OUT}/logs/{prefix}_{tag}_checkpoints.parquet"
            if os.path.exists(p):
                out.append((f"{prefix}_{tag}", st))
                break
    return out


def load(RUNS, contrib_pct=20.0):
    fr = []
    for run, stride in RUNS:
        p = f"{OUT}/logs/{run}_checkpoints.parquet"
        if not os.path.exists(p):
            print(f"[skip] missing {p}")
            continue
        d = pd.read_parquet(p)
        d["stride"] = stride
        fr.append(d)
    if len(fr) < 2:
        raise FileNotFoundError("need at least two stride runs")
    df = pd.concat(fr, ignore_index=True)
    df = df[df.gt_valid & (df.B_perp > 1e-4) & (df.err_p2pl > 0) & (df.n_obs >= 2)]
    df = df[df.contrib > np.percentile(df.contrib, contrib_pct)].copy()
    df["age"] = df.frame - df.birth_frame
    df["err_over_scale"] = df.err_p2pl / df.scale_mean.clip(lower=1e-6)
    df["frac_unresolved"] = df.err_radial / df.err_init.clip(lower=1e-4)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrib-pct", type=float, default=20.0)
    ap.add_argument("--prefix", default="nrgbd_staircase")
    args = ap.parse_args()
    RUNS = resolve_runs(args.prefix)
    print(f"[runs] {RUNS}")
    df = load(RUNS, args.contrib_pct)

    print("=== matched-age comparison across frame stride ===")
    print("if parallax drives accuracy, err must fall as stride rises within an age row\n")
    rows = []
    for lo, hi in AGE_BINS:
        s = df[(df.age > lo) & (df.age <= hi)]
        if len(s) < 500:
            continue
        line = {"age": f"({lo},{hi}]"}
        for _, stride in RUNS:
            t = s[s.stride == stride]
            if len(t) < 100:
                continue
            line[f"B@{stride}"] = t.B_perp.median()
            line[f"alpha@{stride}"] = t.alpha_max.median()
            line[f"err_mm@{stride}"] = t.err_radial.median() * 1e3
            line[f"scale_mm@{stride}"] = t.scale_mean.median() * 1e3
            line[f"e/s@{stride}"] = t.err_over_scale.median()
        rows.append(line)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/tables/stride_instrument_{args.prefix}.csv", index=False)
    print(tab.to_string(index=False, float_format="%.3f"))

    # Formal test: log err ~ log B + log z + log scale + age fixed effects.
    # Age FE absorb the staleness channel; identification is then purely
    # cross-stride variation in baseline at equal age.
    d = df[(df.err_radial > 0) & (df.scale_mean > 0)].copy()
    d["abin"] = pd.cut(d.age, [0, 5, 10, 20, 40, 80, 160, 10**6])
    d = d.dropna(subset=["abin"])
    y = np.log(d.err_radial.values)
    Xc = np.stack([np.log(d.B_perp.values), np.log(d.z_mean.values),
                   np.log(d.scale_mean.values)], 1)
    D = pd.get_dummies(d.abin, drop_first=True).values.astype(float)
    X = np.concatenate([np.ones((len(d), 1)), Xc, D], 1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ beta).var() / y.var())

    rng = np.random.default_rng(0)
    keys = (d.run.astype(str) + ":" + d.gid.astype(str)).values
    uniq, inv = np.unique(keys, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    boots = []
    for _ in range(200):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[i] for i in pick])
        try:
            boots.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][1])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n=== age-fixed-effects regression (identification = cross-stride) ===")
    print(f"  b_logB   = {beta[1]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   (theory: -1)")
    print(f"  c_logz   = {beta[2]:+.3f}")
    print(f"  d_lscale = {beta[3]:+.3f}")
    print(f"  R2       = {r2:.3f}   n={len(d):,}  lineages={len(uniq):,}")

    verdict = {
        "prefix": args.prefix,
        "identification": "frame stride as instrument for baseline at matched age",
        "b_logB": float(beta[1]), "b_ci": [float(lo), float(hi)],
        "c_logz": float(beta[2]), "d_logscale": float(beta[3]),
        "r2": float(r2), "n": int(len(d)), "n_lineages": int(len(uniq)),
        "H1_parallax_reduces_error": bool(hi < 0.0),
        "theory_b_minus_1_in_ci": bool(lo <= -1.0 <= hi),
    }
    with open(f"{OUT}/tables/verdict_instrument_{args.prefix}.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== verdict ===")
    print(json.dumps(verdict, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, stride in RUNS:
        s = df[df.stride == stride]
        if len(s) < 500:
            continue
        q = np.percentile(s.age, np.linspace(2, 98, 12))
        for a, col in ((ax[0], "err_radial"), (ax[1], "err_over_scale")):
            xs, ys = [], []
            for j in range(len(q) - 1):
                k = (s.age >= q[j]) & (s.age < q[j + 1])
                if k.sum() >= 50:
                    xs.append(np.median(s.age[k]))
                    ys.append(np.median(s[col][k]))
            a.plot(xs, ys, "-o", ms=4, label=f"stride {stride}")
    ax[0].set_ylabel("median err_radial [m]")
    ax[1].set_ylabel("median err_p2pl / scale")
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("age [stream frames]")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("Matched age, 1x/2x/4x accumulated baseline: does parallax help?", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figs/fig3_stride_{args.prefix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
