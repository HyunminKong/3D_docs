"""Does descriptor proximity predict that a borrowed skill will not hurt?

Reads the pairwise penalties and asks three things in order of how damaging a
negative answer would be:

1. Is a state's own correction gentler than a stranger's?  If not, the states
   carry nothing specific and there is nothing to retrieve.
2. Among strangers, does descriptor distance track the damage?  This is the
   retrieval question proper: a bank keyed on a descriptor can only work if
   near neighbours are more interchangeable than far ones.
3. Does clustering the descriptor separate compatible from incompatible states,
   and at what number of clusters does that stop improving?

Penalties are compared as magnitudes, and the descriptor is restricted to the
causal features -- the ones a stream could compute before the segment it
describes has finished arriving.
"""

import argparse
import itertools
import json

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

FEATURES = ["causal_prefix25", "causal_spread_a", "causal_rot_a", "causal_arc_a"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="oracle/results/retrieval_sim.json")
    parser.add_argument("--features", default=",".join(FEATURES))
    parser.add_argument("--ks", default="2,3,4,5,6,8,10")
    args = parser.parse_args()

    data = json.load(open(args.results))
    units = data["units"]
    ids = [u["id"] for u in units]
    index = {u["id"]: i for i, u in enumerate(units)}
    feats = [f for f in args.features.split(",") if f]

    x = np.array([[u["desc"][f] for f in feats] for u in units], float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    dist = np.linalg.norm(x[:, None] - x[None], axis=-1)

    pairs = data["pairs"]
    delta = np.full((len(units), len(units)), np.nan)
    for p in pairs:
        delta[index[p["target"]], index[p["source"]]] = p["delta_a"]

    self_p = np.array([delta[i, i] for i in range(len(units))])
    off = ~np.eye(len(units), dtype=bool)
    other_p = delta[off]
    same_scene = np.array([[units[i]["scene"] == units[j]["scene"]
                            for j in range(len(units))] for i in range(len(units))])

    print(f"{len(units)} units, bank layers {data['bank_layers']}, "
          f"rank {data['rank']}, lambda {data['lam']}")
    print(f"features: {', '.join(feats)}\n")

    print("1) own vs borrowed correction, mean penalty in dB")
    print(f"   self                    {self_p.mean():+.3f}   (n={len(self_p)})")
    cross_same = delta[same_scene & off]
    cross_diff = delta[~same_scene]
    print(f"   same scene, other cut   {cross_same.mean():+.3f}   (n={cross_same.size})")
    print(f"   different scene         {cross_diff.mean():+.3f}   (n={cross_diff.size})")
    wins = sum(1 for i in range(len(units))
               if self_p[i] > np.nanmean(delta[i][np.arange(len(units)) != i]))
    print(f"   own is gentler in {wins}/{len(units)} units")
    u_stat = stats.mannwhitneyu(self_p, other_p, alternative="greater")
    print(f"   Mann-Whitney self > borrowed: p = {u_stat.pvalue:.2e}")

    print("\n2) does descriptor distance predict the damage? (borrowed pairs only)")
    d_off, p_off = dist[off], delta[off]
    rho, pval = stats.spearmanr(d_off, p_off)
    print(f"   all borrowed pairs        rho = {rho:+.3f}  p = {pval:.2e}  (n={d_off.size})")
    mask = (~same_scene)[off]
    rho_d, p_d = stats.spearmanr(d_off[mask], p_off[mask])
    print(f"   different-scene pairs     rho = {rho_d:+.3f}  p = {p_d:.2e}  (n={mask.sum()})")
    # nearest neighbour vs random source
    nn_gain = []
    for i in range(len(units)):
        others = [j for j in range(len(units)) if j != i]
        nn = min(others, key=lambda j: dist[i, j])
        nn_gain.append(delta[i, nn] - np.mean([delta[i, j] for j in others]))
    nn_gain = np.array(nn_gain)
    print(f"   nearest-neighbour source vs average source: {nn_gain.mean():+.3f} dB "
          f"({(nn_gain > 0).sum()}/{len(units)} units better)")

    print("\n3) clustering the descriptor")
    print(f"   {'K':>3} {'within':>9} {'between':>9} {'gap':>8}")
    for k in [int(v) for v in args.ks.split(",")]:
        if k >= len(units):
            continue
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
        same_cluster = labels[:, None] == labels[None]
        within = delta[same_cluster & off]
        between = delta[~same_cluster & off]
        if within.size == 0 or between.size == 0:
            continue
        print(f"   {k:>3} {within.mean():+9.3f} {between.mean():+9.3f} "
              f"{within.mean() - between.mean():+8.3f}")

    print("\ninterpretation")
    if u_stat.pvalue < 0.05 and rho < -0.15:
        print("  states are specific AND descriptor distance tracks compatibility:")
        print("  retrieval on this key is supported")
    elif u_stat.pvalue < 0.05:
        print("  states are specific, but descriptor distance does not track")
        print("  compatibility: the key is wrong, not the idea")
    else:
        print("  own correction is no gentler than a stranger's: nothing to retrieve")


if __name__ == "__main__":
    main()
