#!/usr/bin/env python3
"""Create a larger scene-disjoint development validation split without test leakage.

The original manifest deliberately has a small validation component.  This
utility moves whole *training* overlap components into validation while
retaining every original test record unchanged.  It must be run before any
bootstrap or model-selection training that uses the resulting manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def components(records: list[dict]) -> list[set[str]]:
    nodes = {scene for row in records for scene in (row["source_scene"], row["target_scene"])}
    parent = {node: node for node in nodes}
    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for row in records:
        left, right = find(row["source_scene"]), find(row["target_scene"])
        if left != right:
            parent[left] = right
    groups: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        groups[find(node)].add(node)
    return list(groups.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-components", type=int, default=1,
                        help="number of largest original-train components moved to development validation")
    args = parser.parse_args()
    records = json.loads(Path(args.source).read_text())
    train = [row for row in records if row["split"] == "train"]
    groups = components(train)
    groups.sort(key=lambda group: (-sum(row["source_scene"] in group for row in train), sorted(group)))
    heldout_scenes = set().union(*groups[:args.holdout_components])
    output = []
    for row in records:
        row = dict(row)
        if row["split"] == "train" and row["source_scene"] in heldout_scenes:
            row["split"] = "val"
        output.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    summary = {split: sum(row["split"] == split for row in output) for split in ("train", "val", "test")}
    print(json.dumps({"out": args.out, "summary": summary,
                      "new_validation_scenes": sorted(heldout_scenes), "test_unchanged": True}))


if __name__ == "__main__":
    main()
