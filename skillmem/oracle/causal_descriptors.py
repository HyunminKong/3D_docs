"""Can the descriptor be computed before the thing it describes has happened?

`parallax_ratio` -- the arc B travels divided by the extent A covered -- came out
strongest within scenes, but it reads the whole B segment. At the moment a skill
would actually be retrieved, B has not arrived. Its predictive power may be
entirely a consequence of seeing the future.

Every recorded property is first labelled by what it reads. Those built only
from A are causal as they stand and give a lower bound on what a legitimate
descriptor can do. The rest are replaced by variants that read only a prefix of
B, which is what a stream would have:

  prefix     arc of the first p% of B, over A's extent
  velocity   mean per-view displacement over that prefix, extrapolated to the
             full horizon, over A's extent
  instant    displacement from A's last camera to B's first, over A's extent

Correlations are computed exactly as before -- within scene, partialling out
both `a_span` and the A-only PSNR -- so the numbers sit alongside the
non-causal ones on the same scale.

Poses only, no GPU: the split plan is deterministic, so any geometry variant can
be recomputed for the 280 observations already measured.
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

from oracle.analyze_multisplit import benjamini_hochberg, centre_within
from oracle.oracle_data import SceneViews, segment_plan
from oracle.scene_profile import rotation_angle

# what each recorded property reads
A_ONLY = {"arc_a", "spread_a", "baseline_a", "rot_a_deg"}
READS_B = {
    "arc_b", "spread_b", "baseline_b", "rot_b_deg", "arc_ratio", "parallax_ratio",
    "ab_centroid_dist", "ab_min_dist", "ab_mean_dist", "b_closest_to_a",
    "ab_view_angle_deg", "ab_mean_pair_angle_deg", "ab_min_pair_angle_deg",
}


def arc(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0


def spread(c):
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    return float(d.max()) if len(c) > 1 else 1e-6


def causal_variants(c2w_a, c2w_b, prefixes=(0.25, 0.5)):
    """Descriptors a stream could actually compute at retrieval time."""
    ca, cb = c2w_a[:, :3, 3], c2w_b[:, :3, 3]
    extent = max(spread(ca), 1e-6)
    out = {
        "causal_arc_a": arc(ca),
        "causal_spread_a": spread(ca),
        "causal_rot_a": float(
            sum(rotation_angle(c2w_a[i, :3, :3], c2w_a[i + 1, :3, :3]) for i in range(len(c2w_a) - 1))
        ),
        "causal_instant": float(np.linalg.norm(cb[0] - ca[-1])) / extent,
    }
    n_b = len(cb)
    for p in prefixes:
        k = max(2, int(round(n_b * p)))
        pre = cb[:k]
        out[f"causal_prefix{int(p * 100)}"] = arc(pre) / extent
        steps = np.linalg.norm(np.diff(pre, axis=0), axis=1)
        out[f"causal_velocity{int(p * 100)}"] = float(steps.mean() * (n_b - 1)) / extent
    return out


def partial_two(x, y, z1, z2):
    """Spearman of x,y after removing z1 and z2, on ranks."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([stats.rankdata(z1), stats.rankdata(z2), np.ones(len(x))])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]  # noqa: E731
    return stats.spearmanr(res(rx), res(ry))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="oracle/results/multisplit.json")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--out", default="oracle/results/causal.json")
    args = parser.parse_args()

    rows = json.load(open(args.profiles))
    cache = {}
    for row in rows:
        path = os.path.join(args.processed_dir, row["scene"], "opencv_cameras.json")
        if row["scene"] not in cache:
            cache[row["scene"]] = SceneViews(path, 1, 1)
        scene = cache[row["scene"]]
        plan = segment_plan(
            len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b,
            a_span=row["a_span"], gap=row["gap"],
        )
        norm = scene.normalise(sorted(set(sum(plan.values(), []))))
        c2w = norm["c2w"]
        row.update(causal_variants(
            np.stack([c2w[v].numpy() for v in plan["a_input"]]),
            np.stack([c2w[v].numpy() for v in plan["b_input"]]),
        ))

    scenes = [r["scene"] for r in rows]
    y = centre_within(np.array([r["interference"] for r in rows], float), scenes)
    z1 = centre_within(np.array([r["psnr_a_only"] for r in rows], float), scenes)
    z2 = centre_within(np.array([r["a_span"] for r in rows], float), scenes)

    keys = [k for k in rows[0] if k.startswith("causal_")] + sorted(A_ONLY | READS_B)
    results = []
    for key in keys:
        x = np.array([r[key] for r in rows], float)
        if np.allclose(x, x[0]):
            continue
        rho, p = partial_two(centre_within(x, scenes), y, z1, z2)
        causal = key.startswith("causal_") or key in A_ONLY
        results.append({"key": key, "rho": float(rho), "p": float(p), "causal": causal})

    qs = benjamini_hochberg(np.array([r["p"] for r in results]))
    for row_, q in zip(results, qs):
        row_["q"] = float(q)
    results.sort(key=lambda r: -abs(r["rho"]))

    print(f"n = {len(rows)} observations, {len(set(scenes))} scenes")
    print("partial correlation with interference, within scene, "
          "controlling for a_span and psnr_a_only\n")
    print(f"{'property':>26} {'causal':>7} {'rho':>8} {'q(BH)':>8}")
    for row_ in results:
        tag = "yes" if row_["causal"] else "-"
        mark = " *" if row_["q"] < 0.05 else ""
        print(f"{row_['key']:>26} {tag:>7} {row_['rho']:+8.3f} {row_['q']:8.4f}{mark}")

    causal_rows = [r for r in results if r["causal"] and r["q"] < 0.05]
    best = max(causal_rows, key=lambda r: abs(r["rho"])) if causal_rows else None
    print()
    if best:
        verdict = ("PASS" if abs(best["rho"]) >= 0.30
                   else "WEAK" if abs(best["rho"]) >= 0.15 else "FAIL")
        print(f"best causal: {best['key']}  rho = {best['rho']:+.3f}  -> {verdict}")
    else:
        print("no causal property survives correction -> FAIL")
    ref = next((r for r in results if r["key"] == "parallax_ratio"), None)
    if ref and best:
        print(f"non-causal reference parallax_ratio = {ref['rho']:+.3f}; "
              f"causal retains {abs(best['rho']) / abs(ref['rho']) * 100:.0f}% of it")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
