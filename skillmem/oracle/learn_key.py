"""Is there an interaction for a retrieval key to predict, and can one be learned?

The pairwise penalties showed that a unit's own correction is five times gentler
than a stranger's, but Euclidean distance on the geometric descriptor predicted
nothing (rho = -0.04). Two very different things could produce that.

The penalty could be almost fully explained by main effects -- some sources are
harmful to everybody, some targets are fragile -- in which case the best source
is the same for every query, that is a constant rather than a retrieval, and no
key exists to be found. Or there could be genuine interaction that the hand-
chosen geometry simply fails to capture.

So the matrix is decomposed first:

    penalty(t, s) = mu + target(t) + source(s) + same_scene + interaction

Only if the interaction term carries real variance is it worth learning a key,
and then only under leave-one-unit-out validation: with twenty units, anything
fitted in-sample will look excellent and mean nothing.

A note on what a positive result would mean. If compatibility turns out to be
governed by scene identity rather than by how the camera moved, then what makes
two adaptations interchangeable is the content they encode, and the distinction
from content memory -- the thing this project is supposed to be different from --
gets thinner. That is checked explicitly.
"""

import argparse
import json

import numpy as np
from scipy import stats

CAUSAL = ["causal_prefix25", "causal_prefix50", "causal_spread_a", "causal_rot_a",
          "causal_arc_a", "causal_instant", "causal_velocity25"]


def decompose(delta, same_scene, off):
    """Two-way decomposition of the penalty matrix, ignoring the diagonal."""
    y = delta[off]
    total = y.var()
    n = delta.shape[0]

    grand = y.mean()
    tgt = np.array([np.nanmean(delta[i][np.arange(n) != i]) for i in range(n)]) - grand
    src = np.array([np.nanmean(delta[:, j][np.arange(n) != j]) for j in range(n)]) - grand

    fitted_main = grand + tgt[:, None] + src[None, :]
    resid_main = (delta - fitted_main)[off]

    ss = same_scene[off]
    same_effect = resid_main[ss].mean() - resid_main[~ss].mean() if ss.any() else 0.0
    fitted_scene = fitted_main.copy()
    fitted_scene[same_scene] += resid_main[ss].mean() if ss.any() else 0.0
    fitted_scene[~same_scene] += resid_main[~ss].mean()
    resid_scene = (delta - fitted_scene)[off]

    return {
        "total_var": float(total),
        "explained_by_main": float(1 - resid_main.var() / total),
        "explained_by_main_plus_scene": float(1 - resid_scene.var() / total),
        "same_scene_effect_db": float(same_effect),
        "interaction_var_share": float(resid_scene.var() / total),
        "resid_scene": resid_scene,
        "fitted_scene": fitted_scene,
    }


def loo_key(x, delta, off, ridge=1.0):
    """Leave-one-unit-out: can a linear key on the features rank sources?

    Learns a symmetric bilinear form M so that -(xi - xj)' M (xi - xj)
    tracks the interaction residual, fitting M on all pairs that do not
    involve the held-out unit and scoring the rank correlation on the pairs
    that do.
    """
    n = len(x)
    scores = []
    for held in range(n):
        train = [(i, j) for i in range(n) for j in range(n)
                 if i != j and held not in (i, j)]
        test = [(held, j) for j in range(n) if j != held]
        if not test:
            continue
        def design(pairs):
            rows = []
            for i, j in pairs:
                d = x[i] - x[j]
                rows.append(np.concatenate([np.outer(d, d)[np.triu_indices(len(d))], [1.0]]))
            return np.array(rows)
        a_tr, y_tr = design(train), np.array([delta[i, j] for i, j in train])
        a_te, y_te = design(test), np.array([delta[i, j] for i, j in test])
        w = np.linalg.solve(a_tr.T @ a_tr + ridge * np.eye(a_tr.shape[1]), a_tr.T @ y_tr)
        pred = a_te @ w
        if np.std(pred) < 1e-12:
            continue
        scores.append(stats.spearmanr(pred, y_te).statistic)
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="oracle/results/retrieval_sim.json")
    parser.add_argument("--features", default=",".join(CAUSAL))
    args = parser.parse_args()

    data = json.load(open(args.results))
    units = data["units"]
    index = {u["id"]: i for i, u in enumerate(units)}
    n = len(units)

    delta = np.full((n, n), np.nan)
    for p in data["pairs"]:
        delta[index[p["target"]], index[p["source"]]] = p["delta_a"]
    off = ~np.eye(n, dtype=bool)
    same_scene = np.array([[units[i]["scene"] == units[j]["scene"] for j in range(n)]
                           for i in range(n)])

    dec = decompose(delta, same_scene, off)
    print(f"{n} units, {int(off.sum())} borrowed pairs")
    print(f"penalty variance {dec['total_var']:.4f} dB^2\n")
    print("variance explained")
    print(f"  target + source main effects      {dec['explained_by_main'] * 100:5.1f}%")
    print(f"  + same-scene indicator            {dec['explained_by_main_plus_scene'] * 100:5.1f}%")
    print(f"  left as interaction               {dec['interaction_var_share'] * 100:5.1f}%")
    print(f"  same-scene is worth               {dec['same_scene_effect_db']:+.3f} dB")

    if dec["interaction_var_share"] < 0.15:
        print("\n  main effects dominate: the best source is nearly the same for every\n"
              "  query, which is a constant, not a retrieval")

    print("\nleave-one-unit-out, linear key on causal features")
    feats = [f for f in args.features.split(",") if f]
    x = np.array([[u["desc"][f] for f in feats] for u in units], float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    for label, target in (("raw penalty", delta),
                          ("interaction only", delta - dec["fitted_scene"] + np.nanmean(delta[off]))):
        scores = loo_key(x, target, off)
        if scores.size:
            t = stats.ttest_1samp(scores, 0.0)
            print(f"  {label:>17}: rho = {scores.mean():+.3f} +/- {scores.std():.3f}  "
                  f"(p = {t.pvalue:.3f}, {(scores > 0).sum()}/{scores.size} folds positive)")

    print("\nwhat governs compatibility")
    resid = dec["resid_scene"]
    ss = same_scene[off]
    print(f"  same scene   mean penalty {delta[same_scene & off].mean():+.3f} dB")
    print(f"  other scene  mean penalty {delta[~same_scene].mean():+.3f} dB")
    print(f"  gap {delta[same_scene & off].mean() - delta[~same_scene].mean():+.3f} dB "
          f"vs interaction sd {resid.std():.3f} dB")


if __name__ == "__main__":
    main()
