"""Within-scene descriptor search.

The between-scene analysis failed because interference tracked the A-only PSNR
at rho = +0.83: scenes differ mostly in how reconstructable they are, and that
swamped any geometric effect.  Here every scene contributes several A/B cuts, so
each variable can be centred on its own scene's mean before correlating.  That
removes scene identity exactly -- appearance, texture, difficulty, the lot --
and leaves only how the two segments sit relative to each other, which is the
thing a regime descriptor would have to key on.

Reported alongside:

* how much of the interference variance is within scenes at all.  If cuts of the
  same trajectory all forget the same amount, there is nothing for a descriptor
  to separate and no feature can help.
* the same correlations computed between scenes, so the two views can be
  compared directly.
* p-values corrected across the whole family of candidate properties.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats

EXCLUDE = {"scene", "a_span", "gap", "psnr_a_only", "psnr_a_plus_b", "interference"}


def benjamini_hochberg(pvals):
    order = np.argsort(pvals)
    ranked = np.empty_like(pvals)
    n = len(pvals)
    running = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        running = min(running, pvals[idx] * n / (n - rank + 1))
        ranked[idx] = running
    return ranked


def centre_within(values, scenes):
    """Subtract each scene's own mean, so only within-scene variation remains."""
    means = defaultdict(list)
    for value, scene in zip(values, scenes):
        means[scene].append(value)
    means = {k: np.mean(v) for k, v in means.items()}
    return np.array([v - means[s] for v, s in zip(values, scenes)])


def variance_split(values, scenes):
    """Fraction of total variance that lives within scenes rather than between."""
    groups = defaultdict(list)
    for value, scene in zip(values, scenes):
        groups[scene].append(value)
    grand = np.mean(values)
    within = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values())
    between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    return within / (within + between)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="oracle/results/multisplit.json")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = json.load(open(args.profiles))
    scenes = [r["scene"] for r in rows]
    y = np.array([r["interference"] for r in rows], float)
    n_scenes = len(set(scenes))
    per_scene = len(rows) / max(n_scenes, 1)

    print(f"n = {len(rows)} observations over {n_scenes} scenes ({per_scene:.1f} cuts each)")
    print(f"interference {y.mean():+.3f} +/- {y.std(ddof=1):.3f} dB "
          f"(range {y.min():+.3f} .. {y.max():+.3f})")
    frac = variance_split(y, scenes)
    print(f"within-scene share of interference variance: {frac * 100:.1f}%")
    if frac < 0.2:
        print("  -> cuts of the same scene forget alike; little for a descriptor to key on")

    q_only = np.array([r["psnr_a_only"] for r in rows], float)
    print(f"interference vs psnr_a_only : between {stats.spearmanr(q_only, y)[0]:+.3f}"
          f"   within {stats.spearmanr(centre_within(q_only, scenes), centre_within(y, scenes))[0]:+.3f}")

    yc = centre_within(y, scenes)
    keys = [k for k in rows[0] if k not in EXCLUDE and isinstance(rows[0][k], (int, float))]
    results = []
    for key in keys:
        x = np.array([r[key] for r in rows], float)
        if np.allclose(x, x[0]):
            continue
        rho_w, p_w = stats.spearmanr(centre_within(x, scenes), yc)
        rho_b, p_b = stats.spearmanr(x, y)
        results.append({"key": key, "rho_w": rho_w, "p_w": p_w, "rho_b": rho_b, "p_b": p_b})

    qs = benjamini_hochberg(np.array([r["p_w"] for r in results]))
    for row, q in zip(results, qs):
        row["q_w"] = q
    results.sort(key=lambda r: -abs(r["rho_w"]))

    print(f"\n{'property':>26} {'within':>8} {'q(BH)':>8}   {'between':>8}")
    for row in results:
        mark = " *" if row["q_w"] < 0.05 else ""
        print(f"{row['key']:>26} {row['rho_w']:+8.3f} {row['q_w']:8.4f}{mark}  {row['rho_b']:+8.3f}")

    survivors = [r for r in results if r["q_w"] < 0.05]
    print()
    if survivors:
        print(f"{len(survivors)} propert{'y' if len(survivors) == 1 else 'ies'} survive correction:")
        for row in survivors:
            print(f"  {row['key']}  rho = {row['rho_w']:+.3f}  q = {row['q_w']:.4f}")
    else:
        print("nothing survives correction; no measured geometry predicts interference "
              "once scene identity is removed")


if __name__ == "__main__":
    main()
