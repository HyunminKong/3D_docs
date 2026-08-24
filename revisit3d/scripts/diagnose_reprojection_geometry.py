#!/usr/bin/env python3
"""Check depth scale, field-of-view coverage, and loss degeneracy before TTT.

Known camera transforms are used only as an objective-health counterfactual.
The result tells us whether the selected frame spacing can constrain depth at
all; it is not a deployment metric and is not used for model selection.
"""

from __future__ import annotations

import argparse
import json

import torch

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import reprojection_loss
from revisit3d.scripts.train_oracle_revisit import to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--depths", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    rows = []
    for sample in dataset:
        segment = to_device(sample["a"], "cuda" if torch.cuda.is_available() else "cpu")
        height = width = 16
        measurements = []
        for scalar in args.depths:
            depth = torch.full((1, segment["context"]["rgb"].shape[1], height, width), scalar,
                               device=segment["context"]["rgb"].device)
            loss, stats = reprojection_loss(depth, segment["context"]["rgb"], segment["context"]["intrinsics"],
                                            segment["context"]["w2c"], return_stats=True)
            measurements.append({"depth": scalar, "photometric_loss": float(loss), "valid_fraction": float(stats["valid_fraction"])})
        row = {"episode": sample["episode_id"], "measurements": measurements}
        rows.append(row)
        print(json.dumps(row))
    payload = {"split": args.split, "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
