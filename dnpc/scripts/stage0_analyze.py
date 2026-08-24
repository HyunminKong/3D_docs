"""Stage 0 analysis: does B/z^p collapse the error curves, and at what p?

Primary estimator is a log-log regression rather than the grid sweep:

    log err = a + b log B + c log z + eps        =>   p* = -c / b

This gives p* directly *with a confidence interval*, lets us test whether a single
power law fits at all, and makes the Go/No-Go threshold of RESEARCH_PLAN 6.11
decidable ("p* = 1.7" and "p* = 1.7 +/- 0.6" are different papers). Uncertainty
uses a block bootstrap clustered on birth frame, because Gaussians born in the
same frame share observation geometry and are far from independent.

The p grid sweep and its three collapse statistics are retained as the
*visualisation* (Figure 1) and as a check that the regression optimum agrees with
where the depth-binned curves actually overlap.

    python scripts/stage0_analyze.py --runs nrgbd_staircase_main ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"
P_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# (column, label, is_relative) -- relative variants are what separate the
# "constant absolute precision" (p->2) and "constant relative precision" (p->1)
# error targets, so both must be reported side by side.
ERROR_METRICS = [
    ("err_p2pl", "geometric: point-to-plane [m]", False),
    ("err_p2pl_rel", "geometric: point-to-plane / z", True),
    ("err_radial", "geometric: radial component [m]", False),
    ("err_lateral", "geometric: lateral component [m]", False),
    ("e_depth", "rendered depth error [m]", False),
    ("e_depth_rel", "rendered depth error / z", True),
    ("e_render", "photometric error [0-1]", True),
]


# --------------------------------------------------------------------- load
def load(runs, contrib_pct=20.0, dens_filter="all", min_obs=2):
    frames = []
    for r in runs:
        p = f"{OUT}/logs/{r}_gaussian_stats.parquet"
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)

    df = df[(df.n_obs >= min_obs) & df.gt_valid]
    df = df[(df.B_perp > 1e-4) & (df.z_mean > 1e-3)]
    if contrib_pct > 0:
        thr = np.percentile(df.eval_contrib, contrib_pct)
        df = df[df.eval_contrib > thr]
    if dens_filter == "clean":
        df = df[df.dens_count == 0]

    df = df.copy()
    df["err_p2pl_rel"] = df.err_p2pl / df.z_mean
    df["e_depth_rel"] = df.e_depth / df.z_mean
    print(f"[load] {n0:,} -> {len(df):,} rows "
          f"(contrib>{contrib_pct}pct, dens={dens_filter}, n_obs>={min_obs})")
    return df


# ----------------------------------------------------------------- estimator
def _ols(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    return beta, resid, XtX_inv


def regress_p(df, err_col, n_boot=300, seed=0):
    """p* = -c/b from log err = a + b log B_perp + c log z, with a block bootstrap CI."""
    m = df[err_col].values > 0
    d = df[m]
    if len(d) < 500:
        return None
    y = np.log(d[err_col].values)
    lb, lz = np.log(d.B_perp.values), np.log(d.z_mean.values)
    X = np.stack([np.ones_like(lb), lb, lz], 1)

    beta, resid, _ = _ols(X, y)
    b, c = beta[1], beta[2]
    r2 = 1.0 - resid.var() / y.var()
    p_hat = -c / b if abs(b) > 1e-8 else np.nan

    blocks = d.birth_frame.values
    uniq = np.unique(blocks)
    idx_by_block = {u: np.where(blocks == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_block[u] for u in pick])
        bb, _, _ = _ols(X[sel], y[sel])
        if abs(bb[1]) > 1e-8:
            boots.append(-bb[2] / bb[1])
    boots = np.array(boots)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 10 else (np.nan, np.nan))
    return {
        "metric": err_col, "n": int(len(d)),
        "b_logB": float(b), "c_logz": float(c),
        "p_star": float(p_hat), "p_lo": float(lo), "p_hi": float(hi),
        "p_ci_width": float(hi - lo), "r2_loglog": float(r2),
        "corr_logB_logz": float(np.corrcoef(lb, lz)[0, 1]),
        "sd_logz": float(lz.std()), "sd_logB": float(lb.std()),
    }


# ------------------------------------------------------------- collapse stats
def depth_bins(z, n=5):
    edges = np.percentile(z, np.linspace(0, 100, n + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges, np.digitize(z, edges[1:-1])


def collapse_stats(df, err_col, p, n_bins=5, n_x=14):
    m = df[err_col].values > 0
    d = df[m]
    x = np.log(d.B_perp.values) - p * np.log(d.z_mean.values)
    y = np.log(d[err_col].values)
    _, bid = depth_bins(d.z_mean.values, n_bins)

    lo, hi = np.percentile(x, [5, 95])
    grid = np.linspace(lo, hi, n_x + 1)
    curves = []
    for b in range(n_bins):
        s = bid == b
        if s.sum() < 200:
            continue
        med = np.full(n_x, np.nan)
        for i in range(n_x):
            k = s & (x >= grid[i]) & (x < grid[i + 1])
            if k.sum() >= 30:
                med[i] = np.median(y[k])
        curves.append(med)
    curves = np.array(curves) if curves else np.zeros((0, n_x))

    with np.errstate(invalid="ignore"):
        ok = (~np.isnan(curves)).sum(0) >= 2
        spread = float(np.nanmean(np.nanvar(curves[:, ok], axis=0))) if ok.any() else np.nan

    A = np.stack([np.ones_like(x), x], 1)
    beta, resid, _ = _ols(A, y)
    r2 = float(1.0 - resid.var() / y.var())
    rho = float(spearmanr(x, y).statistic)
    return {"p": float(p), "bin_spread": spread, "r2_single": r2, "spearman": rho,
            "curves": curves, "grid": grid}


# ---------------------------------------------------------------------- AUSE
def ause(conf, err, n_steps=50):
    """Sparsification error area. conf high = trusted. Lower AUSE is better."""
    order_c = np.argsort(conf)  # least confident first
    order_e = np.argsort(-err)  # worst error first (oracle)
    fracs = np.linspace(0, 0.95, n_steps)
    n = len(err)
    cu, co = [], []
    for f in fracs:
        k = int(f * n)
        cu.append(err[order_c[k:]].mean())
        co.append(err[order_e[k:]].mean())
    cu, co = np.array(cu), np.array(co)
    denom = cu[0] if cu[0] > 0 else 1.0
    return float(np.trapz((cu - co) / denom, fracs)), fracs, cu / denom, co / denom


# -------------------------------------------------------------------- figures
def fig_collapse(df, err_col, p_star, path, n_bins=5):
    ps = [0.0, 1.0, round(p_star * 2) / 2 if np.isfinite(p_star) else 2.0]
    labels = ["p = 0  (raw baseline)", "p = 1  (triangulation angle / classical SfM)",
              f"p = {ps[2]:g}  (fitted p*)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    edges, _ = depth_bins(df.z_mean.values, n_bins)
    cmap = plt.get_cmap("viridis")
    for ax, p, lab in zip(axes, ps, labels):
        st = collapse_stats(df, err_col, p, n_bins)
        centers = 0.5 * (st["grid"][:-1] + st["grid"][1:])
        for i, cv in enumerate(st["curves"]):
            ax.plot(centers, cv, "-o", ms=3, color=cmap(i / max(1, n_bins - 1)),
                    label=f"z {edges[i]:.1f}-{edges[i+1]:.1f} m")
        ax.set_title(f"{lab}\nbin spread={st['bin_spread']:.4f}  R^2={st['r2_single']:.3f}")
        ax.set_xlabel(r"$\log\,(B_\perp / z^{p})$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"log {err_col}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Figure 1 - depth-normalised collapse ({err_col})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sweep(sweeps, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    keys = [("bin_spread", "between-bin spread (min = p*)", True),
            ("r2_single", "single-curve $R^2$ (max = p*)", False),
            ("spearman", "|Spearman rho| (max = p*)", False)]
    for ax, (k, title, is_min) in zip(axes, keys):
        for name, tab in sweeps.items():
            v = np.abs(tab[k]) if k == "spearman" else tab[k]
            ax.plot(P_GRID, v, "-o", ms=4, label=name)
            best = P_GRID[np.nanargmin(v)] if is_min else P_GRID[np.nanargmax(v)]
            ax.axvline(best, ls=":", lw=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("normalisation exponent p")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("p sweep - collapse quality vs normalisation exponent", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_ause(df, p_star, err_col, path, n_bins=5):
    edges, bid = depth_bins(df.z_mean.values, n_bins)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    conf_all = np.log(df.B_perp.values) - p_star * np.log(df.z_mean.values)
    err_all = df[err_col].values
    rows = []
    cmap = plt.get_cmap("viridis")
    for b in range(n_bins):
        s = bid == b
        if s.sum() < 500:
            continue
        a, fr, cu, co = ause(conf_all[s], err_all[s])
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(s.sum()), "ause": a})
        ax[0].plot(fr, cu, color=cmap(b / max(1, n_bins - 1)),
                   label=f"z {edges[b]:.1f}-{edges[b+1]:.1f} m (AUSE {a:.3f})")
        ax[0].plot(fr, co, color=cmap(b / max(1, n_bins - 1)), ls="--", alpha=0.4)
    ax[0].set_xlabel("fraction removed (least confident first)")
    ax[0].set_ylabel(f"normalised mean {err_col}")
    ax[0].set_title(f"Sparsification (solid) vs oracle (dashed), p={p_star:.2f}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    aus = [[ause(conf_all[bid == b] if (bid == b).sum() >= 500 else np.zeros(1),
                 err_all[bid == b] if (bid == b).sum() >= 500 else np.ones(1))[0]
            for b in range(n_bins)] for p in [p_star]]
    ax[1].bar([r["bin"] for r in rows], [r["ause"] for r in rows], color="steelblue")
    ax[1].set_xlabel("depth bin [m]")
    ax[1].set_ylabel("AUSE")
    ax[1].set_title("AUSE by depth bin (lower = confidence ranks error better)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def fig_anisotropy(df, path, n_bins=5):
    """Axis A check: is radial error really the dominant, later-resolved component?"""
    d = df[(df.err_radial > 0) & (df.err_lateral > 0) & (np.abs(df.n_dot_v) > 0.3)]
    edges, bid = depth_bins(d.z_mean.values, n_bins)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    for b in range(n_bins):
        s = bid == b
        if s.sum() < 300:
            continue
        a = d.alpha_max.values[s]
        r = (d.err_radial.values / np.clip(d.err_lateral.values, 1e-9, None))[s]
        q = np.percentile(a, np.linspace(0, 100, 11))
        xs, ys = [], []
        for i in range(10):
            k = (a >= q[i]) & (a < q[i + 1])
            if k.sum() >= 30:
                xs.append(np.median(a[k]))
                ys.append(np.median(r[k]))
        ax[0].plot(xs, ys, "-o", ms=3, color=cmap(b / max(1, n_bins - 1)),
                   label=f"z {edges[b]:.1f}-{edges[b+1]:.1f} m")
    ax[0].axhline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\alpha_{max}$ [rad]")
    ax[0].set_ylabel("median  err_radial / err_lateral")
    ax[0].set_title("Anisotropy vs accumulated parallax\n(>1 means radial dominates)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ax[1].hist(np.log10(np.clip(d.err_radial / np.clip(d.err_lateral, 1e-9, None), 1e-3, 1e3)),
               bins=80, color="steelblue")
    ax[1].axvline(0, color="k", ls=":")
    ax[1].set_xlabel("log10 (err_radial / err_lateral)")
    ax[1].set_title("Distribution of the anisotropy ratio")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--tag", default="stage0")
    ap.add_argument("--contrib-pct", type=float, default=20.0)
    ap.add_argument("--dens-filter", choices=["all", "clean"], default="all")
    ap.add_argument("--primary", default="err_p2pl")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    runs = args.runs
    if runs is None:
        runs = sorted({os.path.basename(p).replace("_gaussian_stats.parquet", "")
                       for p in glob.glob(f"{OUT}/logs/*_gaussian_stats.parquet")
                       if "smoke" not in p})
    print(f"[runs] {runs}")
    df = load(runs, args.contrib_pct, args.dens_filter)

    # ---- primary estimator
    reg = [r for r in (regress_p(df, m, args.n_boot) for m, _, _ in ERROR_METRICS) if r]
    reg_df = pd.DataFrame(reg)
    reg_df.to_csv(f"{OUT}/tables/p_star_regression.csv", index=False)
    print("\n=== p* from log-log regression (block bootstrap CI on birth frame) ===")
    print(reg_df[["metric", "n", "p_star", "p_lo", "p_hi", "p_ci_width",
                  "r2_loglog", "b_logB", "corr_logB_logz"]].to_string(index=False,
                                                                     float_format="%.3f"))

    # ---- grid sweep (visualisation + agreement check)
    sweeps, rows = {}, []
    for m, _, _ in ERROR_METRICS:
        if (df[m].values > 0).sum() < 500:
            continue
        tab = {k: [] for k in ("bin_spread", "r2_single", "spearman")}
        for p in P_GRID:
            st = collapse_stats(df, m, p)
            for k in tab:
                tab[k].append(st[k])
            rows.append({"metric": m, "p": p, **{k: st[k] for k in tab}})
        sweeps[m] = {k: np.array(v) for k, v in tab.items()}
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(f"{OUT}/tables/p_sweep.csv", index=False)

    primary = args.primary
    pr = reg_df[reg_df.metric == primary]
    p_star = float(pr.p_star.iloc[0]) if len(pr) else 2.0
    sw = sweeps[primary]
    p_min_spread = float(P_GRID[np.nanargmin(sw["bin_spread"])])
    print(f"\n[agreement] regression p*={p_star:.2f}   grid argmin(bin_spread)={p_min_spread:.1f}")

    os.makedirs(f"{OUT}/figs", exist_ok=True)
    fig_collapse(df, primary, p_star, f"{OUT}/figs/fig1_collapse.png")
    fig_sweep(sweeps, f"{OUT}/figs/p_sweep_metrics.png")
    ause_rows = fig_ause(df, p_star, primary, f"{OUT}/figs/ause_by_depth_bin.png")
    fig_anisotropy(df, f"{OUT}/figs/fig2_anisotropy.png")
    pd.DataFrame(ause_rows).to_csv(f"{OUT}/tables/ause_by_depth_bin.csv", index=False)

    # ---- H1: does normalisation actually reduce between-bin spread?
    s0 = sw["bin_spread"][0]
    sbest = np.nanmin(sw["bin_spread"])
    drop = 1.0 - sbest / s0 if s0 > 0 else np.nan
    verdict = {
        "primary_metric": primary,
        "p_star": p_star,
        "p_ci": [float(pr.p_lo.iloc[0]), float(pr.p_hi.iloc[0])] if len(pr) else None,
        "grid_p_min_spread": p_min_spread,
        "bin_spread_p0": float(s0),
        "bin_spread_best": float(sbest),
        "spread_reduction": float(drop),
        "H1_collapse_ge_40pct": bool(drop >= 0.40),
        "H2_p_star_gt_1": bool(len(pr) and pr.p_lo.iloc[0] > 1.0),
        "identifiable": bool(len(pr) and pr.p_ci_width.iloc[0] < 1.0),
    }
    with open(f"{OUT}/tables/verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== verdict ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
