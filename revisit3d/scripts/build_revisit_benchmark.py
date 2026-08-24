#!/usr/bin/env python3
"""Create a pose-verified cross-episode A→B→A' manifest."""

import argparse
from pathlib import Path

from revisit3d.data import RevisitBenchmark, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="converted scene root")
    parser.add_argument("--selection", default="", help="optional nuScenes location metadata JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overlap-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = build_manifest(
        args.root, selection=args.selection or None, overlap_m=args.overlap_m, seed=args.seed
    )
    benchmark = RevisitBenchmark(episodes)
    benchmark.write(args.out)
    print(f"wrote {Path(args.out)}: {len(episodes)} directional episodes, {benchmark.summary()}")


if __name__ == "__main__":
    main()

