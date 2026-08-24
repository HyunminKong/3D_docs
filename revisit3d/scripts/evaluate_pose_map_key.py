#!/usr/bin/env python3
"""Oracle pose-map retrieval upper bound on the development revisit bank."""
from __future__ import annotations

import argparse
import json

import torch

from revisit3d.data import RevisitEpisodeDataset


def centers(segment):
    c2w = torch.linalg.inv(segment["context"]["w2c"])
    return c2w[:, :3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="val", image_size=(224, 224))
    source = [centers(sample["a"]) for sample in dataset]
    target = [centers(sample["a_prime"]) for sample in dataset]
    rows = []
    for index, query in enumerate(target):
        # A persistent map query uses its nearest stored camera/map anchor.
        distance = torch.tensor([torch.cdist(query, candidate).min() for candidate in source])
        rank = int((distance.argsort() == index).nonzero()[0]) + 1
        rows.append({"rank":rank, "positive_distance_m":float(distance[index]),
                     "best_distance_m":float(distance.min())})
    summary={"recall_at_1":sum(r["rank"]==1 for r in rows)/len(rows),
             "recall_at_3":sum(r["rank"]<=3 for r in rows)/len(rows),
             "mean_rank":sum(r["rank"] for r in rows)/len(rows),
             "mean_positive_distance_m":sum(r["positive_distance_m"] for r in rows)/len(rows)}
    with open(args.out,"w") as handle: json.dump({"summary":summary,"rows":rows},handle,indent=2)
    print(json.dumps(summary))

if __name__ == "__main__": main()
