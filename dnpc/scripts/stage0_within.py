"""Stage 0 primary analysis: within-Gaussian identification of the B/z law.

Why not the between-Gaussian regression the plan specifies (6.6/6.7): in these
sequences the camera revisits everything, so every *surviving* Gaussian ends up
observed from almost the whole trajectory. Final-state accumulated baseline has
little spread (alpha_max median ~80 deg), and what spread remains is selection --
a Gaussian with few observations is one on an occlusion boundary or the image
border, which independently carries high error. Between-Gaussian regression
therefore measures selection, not triangulation.

The checkpoint log follows the *same* lineage as its baseline grows, so the
within-lineage variation in B is real and uncontaminated. Two further controls
matter, both discovered in the pilot:

  * **Gaussian extent.** err_p2pl tracks the Gaussian's own scale almost 1:1
    (err/scale ~ 1.0-1.4 in every scale bin), and scale is set at spawn to 2z/f.
    A large part of any apparent z-dependence is that identity, not geometry.
    Scale therefore enters as a covariate, and a scale-normalised target is
    reported alongside.
  * **Initial uncertainty.** Each Gaussian spawns with a known depth error
    (err_init). The quantity information sufficiency should predict is the
    *fraction resolved*, err/err_init -- dimensionless and scale-free.

Estimates use lineage fixed effects (within transform) with a cluster bootstrap
over lineages.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "/home/khm/3D_4D/dnpc/outputs/stage0"

TARGETS = [
    ("err_p2pl", "point-to-plane [m]"),
    ("err_radial", "radial component [m]"),
    ("err_lateral", "lateral component [m]"),
    ("err_over_scale", "point-to-plane / Gaussian scale"),
    ("err_resolved", "err_radial / err_init  (fraction unresolved)"),
]


def load_checkpoints(runs, contrib_pct=20.0, min_ckpt=3):
    fr = []
    for r in runs:
        p = f"{OUT}/logs/{r}_checkpoints.parquet"
        if os.path.exists(p):
            fr.append(pd.read_parquet(p))
    if not fr:
        raise FileNotFoundError("no checkpoint logs found")
    df = pd.concat(fr, ignore_index=True)
    n0 = len(df)

    df = df[df.gt_valid & (df.B_perp > 1e-4) & (df.z_mean > 1e-3) & (df.err_p2pl > 0)]
    df = df[df.n_obs >= 2]
    if contrib_pct > 0:
        df = df[df.contrib > np.percentile(df.contrib, contrib_pct)]
    df = df.copy()
    df["key"] = df["run"].astype(str) + ":" + df["gid"].astype(str)
    cnt = df.groupby("key")["frame"].transform("size")
    df = df[cnt >= min_ckpt].copy()

    df["age"] = df.frame - df.birth_frame
    df["err_over_scale"] = df.err_p2pl / df.scale_mean.clip(lower=1e-6)
    df["err_resolved"] = df.err_radial / df.err_init.clip(lower=1e-4)
    print(f"[load] {n0:,} -> {len(df):,} rows, {df.key.nunique():,} lineages "
          f"(>= {min_ckpt} checkpoints each)")
    return df


def within(df, cols):
    """Subtract the per-lineage mean: the fixed-effects transform."""
    g = df.groupby("key")
    return {c: (df[c] - g[c].transform("mean")).values for c in cols}


def fe_regress(df, target, n_boot=200, seed=0):
    d = df[df[target] > 0].copy()
    if len(d) < 1000:
        return None
    for c, src in (("ly", target), ("lB", "B_perp"), ("lz", "z_mean"),
                   ("ls", "scale_mean")):
        d[c] = np.log(d[src].clip(lower=1e-9))
    w = within(d, ["ly", "lB", "lz", "ls"])

    def fit(idx, use_scale):
        cols = [w["lB"][idx], w["lz"][idx]] + ([w["ls"][idx]] if use_scale else [])
        X = np.stack(cols, 1)
        y = w["ly"][idx]
        b = np.linalg.lstsq(X, y, rcond=None)[0]  # no intercept: already demeaned
        r2 = 1.0 - ((y - X @ b).var() / y.var()) if y.var() > 0 else np.nan
        return b, r2

    n = len(d)
    all_idx = np.arange(n)
    out = {}
    keys = d.key.values
    uniq, inv = np.unique(keys, return_inverse=True)
    idx_by_key = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)

    for use_scale in (False, True):
        b, r2 = fit(all_idx, use_scale)
        boots_b, boots_c = [], []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            sel = np.concatenate([idx_by_key[i] for i in pick])
            try:
                bb, _ = fit(sel, use_scale)
            except np.linalg.LinAlgError:
                continue
            boots_b.append(bb[0])
            boots_c.append(bb[1])
        tag = "ctrl" if use_scale else "raw"
        out[f"b_logB_{tag}"] = float(b[0])
        out[f"c_logz_{tag}"] = float(b[1])
        out[f"r2_{tag}"] = float(r2)
        if boots_b:
            out[f"b_lo_{tag}"], out[f"b_hi_{tag}"] = [
                float(v) for v in np.percentile(boots_b, [2.5, 97.5])]
            out[f"c_lo_{tag}"], out[f"c_hi_{tag}"] = [
                float(v) for v in np.percentile(boots_c, [2.5, 97.5])]
        if use_scale:
            out["d_logscale"] = float(b[2])
    out.update(metric=target, n=int(len(d)), n_lineage=int(len(uniq)))
    return out


def fig_age(df, path):
    """The plan's Figure 1, recast: error vs accumulated parallax, by depth bin,
    tracking the same lineages over time."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    edges = np.percentile(df.z_mean, [0, 25, 50, 75, 100])
    bid = np.digitize(df.z_mean, edges[1:-1])
    cmap = plt.get_cmap("viridis")
    for ax, (tgt, lab) in zip(axes, [("err_p2pl", "point-to-plane [m]"),
                                     ("err_radial", "radial [m]"),
                                     ("err_resolved", "err_radial / err_init")]):
        d = df[df[tgt] > 0]
        b2 = np.digitize(d.z_mean, edges[1:-1])
        for i in range(4):
            s = d[b2 == i]
            if len(s) < 300:
                continue
            q = np.percentile(s.alpha_max, np.linspace(2, 98, 12))
            xs, ys = [], []
            for j in range(len(q) - 1):
                k = (s.alpha_max >= q[j]) & (s.alpha_max < q[j + 1])
                if k.sum() >= 50:
                    xs.append(np.median(s.alpha_max[k]))
                    ys.append(np.median(s[tgt][k]))
            ax.plot(xs, ys, "-o", ms=3, color=cmap(i / 3),
                    label=f"z {edges[i]:.1f}-{edges[i+1]:.1f} m")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"accumulated $\alpha_{max}$ [rad]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 1 - within-lineage error vs accumulated parallax, by depth", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scale_confound(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    d = df[(df.err_p2pl > 0) & (df.scale_mean > 0)]
    ax[0].hexbin(np.log10(d.scale_mean), np.log10(d.err_p2pl), gridsize=60,
                 bins="log", cmap="magma")
    lim = [np.log10(d.scale_mean).min(), np.log10(d.scale_mean).max()]
    ax[0].plot(lim, lim, "w--", lw=1.5, label="err = scale")
    ax[0].set_xlabel("log10 Gaussian scale [m]")
    ax[0].set_ylabel("log10 err_p2pl [m]")
    ax[0].set_title(f"Scale confound (corr={np.corrcoef(np.log(d.scale_mean), np.log(d.err_p2pl))[0,1]:+.2f})")
    ax[0].legend(fontsize=8)

    q = np.percentile(d.alpha_max, np.linspace(2, 98, 14))
    for tgt, lab in (("err_p2pl", "err_p2pl"), ("err_over_scale", "err / scale")):
        xs, ys = [], []
        for j in range(len(q) - 1):
            k = (d.alpha_max >= q[j]) & (d.alpha_max < q[j + 1])
            if k.sum() >= 50:
                xs.append(np.median(d.alpha_max[k]))
                ys.append(np.median(d[tgt][k]) / np.median(d[tgt]))
        ax[1].plot(xs, ys, "-o", ms=4, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\alpha_{max}$ [rad]")
    ax[1].set_ylabel("median, normalised")
    ax[1].set_title("Does parallax help once scale is removed?")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--contrib-pct", type=float, default=20.0)
    args = ap.parse_args()

    runs = args.runs or sorted(
        os.path.basename(p).replace("_checkpoints.parquet", "")
        for p in glob.glob(f"{OUT}/logs/*_checkpoints.parquet"))
    print(f"[runs] {runs}")
    df = load_checkpoints(runs, args.contrib_pct)

    print("\n=== age-stratified medians (the raw phenomenon) ===")
    df["_ab"] = pd.cut(df.age, [-1, 5, 15, 30, 60, 120, 10**6])
    print(df.groupby("_ab", observed=True).agg(
        n=("err_p2pl", "size"), B_perp=("B_perp", "median"),
        alpha=("alpha_max", "median"), z=("z_mean", "median"),
        scale_mm=("scale_mean", lambda v: v.median() * 1e3),
        err_p2pl_mm=("err_p2pl", lambda v: v.median() * 1e3),
        err_radial_mm=("err_radial", lambda v: v.median() * 1e3),
        err_over_scale=("err_over_scale", "median"),
        frac_unresolved=("err_resolved", "median"),
    ).to_string(float_format="%.3f"))

    rows = [r for r in (fe_regress(df, t, args.n_boot) for t, _ in TARGETS) if r]
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/tables/within_lineage_regression.csv", index=False)
    print("\n=== lineage fixed-effects regression (cluster bootstrap 95% CI) ===")
    print("theory for a triangulation-limited error: b_logB = -1, c_logz = +2\n")
    for _, r in res.iterrows():
        print(f"  {r.metric:16s} n={r.n:8,} lineages={r.n_lineage:7,}")
        print(f"     raw       b_logB={r.b_logB_raw:+.3f} [{r.b_lo_raw:+.3f},{r.b_hi_raw:+.3f}]"
              f"  c_logz={r.c_logz_raw:+.3f} [{r.c_lo_raw:+.3f},{r.c_hi_raw:+.3f}]  R2={r.r2_raw:.3f}")
        print(f"     +scale    b_logB={r.b_logB_ctrl:+.3f} [{r.b_lo_ctrl:+.3f},{r.b_hi_ctrl:+.3f}]"
              f"  c_logz={r.c_logz_ctrl:+.3f} [{r.c_lo_ctrl:+.3f},{r.c_hi_ctrl:+.3f}]  R2={r.r2_ctrl:.3f}"
              f"  d_logscale={r.d_logscale:+.3f}")

    os.makedirs(f"{OUT}/figs", exist_ok=True)
    fig_age(df, f"{OUT}/figs/fig1_within_lineage.png")
    fig_scale_confound(df, f"{OUT}/figs/fig2_scale_confound.png")

    prim = res[res.metric == "err_radial"].iloc[0]
    verdict = {
        "design": "within-lineage fixed effects (between-Gaussian is confounded by selection)",
        "primary_metric": "err_radial",
        "b_logB": prim.b_logB_ctrl, "b_ci": [prim.b_lo_ctrl, prim.b_hi_ctrl],
        "c_logz": prim.c_logz_ctrl, "c_ci": [prim.c_lo_ctrl, prim.c_hi_ctrl],
        "theory_b": -1.0, "theory_c": 2.0,
        "b_consistent_with_theory": bool(prim.b_lo_ctrl <= -1.0 <= prim.b_hi_ctrl),
        "b_significantly_negative": bool(prim.b_hi_ctrl < 0.0),
        "r2_within": prim.r2_ctrl,
        "n_lineages": int(prim.n_lineage),
    }
    with open(f"{OUT}/tables/verdict_within.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== verdict ===")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
