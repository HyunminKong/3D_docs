#!/usr/bin/env python3
"""Report the spectrum of realised compact TTT updates, without a memory bank."""

from __future__ import annotations

import argparse
import json

import torch


def spectrum(matrix: torch.Tensor) -> dict:
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    energy = singular.square()
    total = energy.sum().clamp_min(1e-12)
    cumulative = energy.cumsum(0) / total
    ranks = {str(rank): float(cumulative[min(rank, len(cumulative)) - 1]) for rank in (1, 2, 4, 8, 16, 32)}
    effective_rank = float(torch.exp(-(energy / total * (energy / total).clamp_min(1e-12).log()).sum()))
    return {"singular_values": singular.tolist(), "explained_energy": ranks, "effective_rank": effective_rank}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    item = torch.load(args.vectors, map_location="cpu", weights_only=False)
    combined = torch.cat((item["a"], item["a_prime"]), dim=0).float()
    output = {"vectors": args.vectors, "n_updates": int(combined.shape[0]), "state_dim": int(combined.shape[1]),
              "spectrum": spectrum(combined)}
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
