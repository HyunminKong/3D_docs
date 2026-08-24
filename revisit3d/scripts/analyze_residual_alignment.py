#!/usr/bin/env python3
"""Test whether context specificity appears after removing global TTT modes."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F


def average_cosine(left: torch.Tensor, right: torch.Tensor, pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    values = [F.cosine_similarity(left[i], right[j], dim=0, eps=1e-8) for i, j in pairs]
    return float(torch.stack(values).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(0, 1, 2, 4, 8))
    args = parser.parse_args()
    item = torch.load(args.vectors, map_location="cpu", weights_only=False)
    a, ap = item["a"].float(), item["a_prime"].float()
    combined = torch.cat((a, ap), dim=0)
    mean = combined.mean(0, keepdim=True)
    _, _, right = torch.linalg.svd(combined - mean, full_matrices=False)
    matched = [(i, i) for i in range(len(a))]
    foreign = [(i, j) for i in range(len(a)) for j in range(len(ap))
               if i != j and not set(item["scene_sets"][i]).intersection(item["scene_sets"][j])]
    rows = []
    for rank in args.ranks:
        basis = right[:rank]
        def residual(x):
            x = x - mean
            return x if rank == 0 else x - (x @ basis.T) @ basis
        ra, rap = residual(a), residual(ap)
        match = average_cosine(ra, rap, matched)
        other = average_cosine(ra, rap, foreign)
        rows.append({"removed_global_rank": rank, "matched_cosine": match, "foreign_cosine": other,
                     "matched_minus_foreign": None if other is None else match - other,
                     "mean_residual_norm": float(torch.cat((ra, rap)).norm(dim=-1).mean())})
    output = {"vectors": args.vectors, "n_matched": len(matched), "n_foreign": len(foreign), "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
