#!/usr/bin/env python3
"""Verify that the benchmark's matched revisits are separable in frozen context.

Physical centre overlap alone is insufficient for the research claim: a
revisit must also be more similar under the available foundation context than
the intervening and scene-disjoint controls.  This script does not train or
adapt anything.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.scripts.train_oracle_revisit import to_device


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=0, eps=1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").to(device)

    descriptors, scene_sets, rows = [], [], []
    for sample in dataset:
        descriptor = []
        for tag in ("a", "b", "a_prime"):
            segment = to_device(sample[tag], device)
            feature = extractor(segment["context"]["rgb"]).mean(dim=(0, 1, 2))
            descriptor.append(F.normalize(feature, dim=0))
        a, b, ap = descriptor
        rows.append({"episode": sample["episode_id"], "matched_cosine": cosine(a, ap),
                     "intervening_cosine": cosine(a, b)})
        descriptors.append((a, ap))
        scene_sets.append({sample["a"]["scene"], sample["a_prime"]["scene"]})
        print(json.dumps(rows[-1]))
    foreign = [cosine(source, target) for i, (source, _) in enumerate(descriptors)
               for j, (_, target) in enumerate(descriptors)
               if i != j and not scene_sets[i].intersection(scene_sets[j])]
    summary = {
        "matched_cosine": sum(row["matched_cosine"] for row in rows) / len(rows),
        "intervening_cosine": sum(row["intervening_cosine"] for row in rows) / len(rows),
        "foreign_cosine": sum(foreign) / len(foreign) if foreign else None,
        "matched_minus_intervening": sum(row["matched_cosine"] - row["intervening_cosine"] for row in rows) / len(rows),
        "matched_minus_foreign": (sum(row["matched_cosine"] for row in rows) / len(rows) - sum(foreign) / len(foreign)) if foreign else None,
    }
    with open(args.out, "w") as handle:
        json.dump({"split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
