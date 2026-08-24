"""Which measurable property of a segment pair predicts how much gets forgotten?

Reads the per-scene profiles and ranks every recorded property by how well it
tracks interference.  Spearman rather than Pearson: the sample is small, the
relationships need not be linear, and a couple of scenes sit far from the rest.

Two guards against reading noise as structure:

* p-values are corrected across the whole family of properties tested
  (Benjamini-Hochberg), because testing twenty candidates against one outcome
  will produce a "significant" one by chance,
* the properties are heavily collinear (arc length, spread and baseline all
  measure how much the camera moved), so the surviving ones are also reported
  after partialling out the single strongest, which is what decides whether a
  descriptor needs one term or several.
"""

import argparse
import json

import numpy as np
from scipy import stats

EXCLUDE = {"scene", "n_frames", "psnr_a_only", "psnr_a_plus_b", "interference"}


def benjamini_hochberg(pvals):
    order = np.argsort(pvals)
    ranked = np.empty_like(pvals)
    n = len(pvals)
    running = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        running = min(running, pvals[idx] * n / (n - rank + 1))
        ranked[idx] = running
    return ranked


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing z, on ranks."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    def residual(v):
        slope, intercept, *_ = stats.linregress(rz, v)
        return v - (slope * rz + intercept)
    return stats.spearmanr(residual(rx), residual(ry))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="oracle/results/profiles.json")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    rows = json.load(open(args.profiles))
    y = np.array([r["interference"] for r in rows], dtype=float)
    n = len(rows)
    keys = [k for k in rows[0] if k not in EXCLUDE and isinstance(rows[0][k], (int, float))]

    print(f"n = {n} scenes   interference {y.mean():+.3f} +/- {y.std(ddof=1):.3f} dB "
          f"(range {y.min():+.3f} .. {y.max():+.3f})\n")

    results = []
    for key in keys:
        x = np.array([r[key] for r in rows], dtype=float)
        if np.allclose(x, x[0]):
            continue
        rho, p = stats.spearmanr(x, y)
        results.append({"key": key, "rho": float(rho), "p": float(p), "x": x})

    pvals = np.array([r["p"] for r in results])
    qvals = benjamini_hochberg(pvals)
    for row, q in zip(results, qvals):
        row["q"] = float(q)
    results.sort(key=lambda r: -abs(r["rho"]))

    print(f"{'property':>26} {'rho':>7} {'p':>8} {'q(BH)':>8}")
    for row in results:
        mark = " *" if row["q"] < 0.05 else ""
        print(f"{row['key']:>26} {row['rho']:+7.3f} {row['p']:8.4f} {row['q']:8.4f}{mark}")

    best = results[0]
    print(f"\nstrongest: {best['key']}  rho = {best['rho']:+.3f}  q = {best['q']:.4f}")
    if n < 15:
        print(f"note: with n = {n}, |rho| must exceed about "
              f"{1.96 / np.sqrt(n - 3):.2f} (Fisher) to be distinguishable from zero")

    print(f"\npartial correlations, controlling for {best['key']}:")
    print(f"{'property':>26} {'rho|z':>8} {'p':>8}")
    for row in results[1 : args.top + 1]:
        rho, p = partial_spearman(row["x"], y, best["x"])
        print(f"{row['key']:>26} {rho:+8.3f} {p:8.4f}")

    lo, hi = np.percentile(y, [33, 67])
    print(f"\nscenes split by interference (low < {lo:.2f} < mid < {hi:.2f} < high):")
    print(f"{'property':>26} {'low':>9} {'mid':>9} {'high':>9}")
    for row in results[: args.top]:
        x = row["x"]
        groups = [x[y <= lo], x[(y > lo) & (y < hi)], x[y >= hi]]
        cells = " ".join(f"{g.mean():9.3f}" if len(g) else f"{'-':>9}" for g in groups)
        print(f"{row['key']:>26} {cells}")


if __name__ == "__main__":
    main()
