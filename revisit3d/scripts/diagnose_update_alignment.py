#!/usr/bin/env python3
"""Measure whether compact-state TTT updates align on true physical revisits.

The online loss uses frozen foundation camera estimates only in this
controlled experiment, keeping depth and pose in a common gauge.  No memory,
retrieval, or residual transport is involved.  It is therefore the direct
test of the first premise: matched geometric contexts should induce more
similar TTT update directions than the intervening or foreign context.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTDepthTeacher, FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, reprojection_loss
from revisit3d.models import StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def adapt_delta(head, extractor, teacher, segment, lr: float, smoothness: float):
    features = extractor(segment["context"]["rgb"])
    grid = int(features.shape[2] ** 0.5)
    geometry = teacher(segment["context"]["rgb"], (grid, grid))
    initial = head.initial_state(1, device=features.device, dtype=features.dtype)
    def objective(prediction):
        depth = depth_grid(prediction)
        return reprojection_loss(depth, segment["context"]["rgb"], geometry["intrinsics"], geometry["w2c"]) + \
            smoothness * depth_smoothness_loss(depth, segment["context"]["rgb"])
    adapted, history = head.adapt(features, initial, objective, steps=1, learning_rate=lr)
    return (adapted.value - initial.value).detach().flatten(), float(history[-1]), geometry


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x, y, dim=0, eps=1e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vectors-out", default="", help="optional .pt path for update-subspace analysis")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled probe requires CUDA")

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    teacher = FrozenVGGTDepthTeacher(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    head.load_state_dict(checkpoint["head"])
    head.eval()

    deltas, rows, scene_sets = [], [], []
    for sample in dataset:
        a, b, ap = (to_device(sample[key], "cuda") for key in ("a", "b", "a_prime"))
        with torch.enable_grad():
            da, la, _ = adapt_delta(head, extractor, teacher, a, args.ttt_lr, args.smoothness)
            db, lb, _ = adapt_delta(head, extractor, teacher, b, args.ttt_lr, args.smoothness)
            dap, lap, _ = adapt_delta(head, extractor, teacher, ap, args.ttt_lr, args.smoothness)
        row = {"episode": sample["episode_id"], "loss_a": la, "loss_b": lb, "loss_a_prime": lap,
               "delta_norm_a": float(da.norm()), "delta_norm_b": float(db.norm()), "delta_norm_a_prime": float(dap.norm()),
               "matched_cosine": cosine(da, dap), "intervening_cosine": cosine(da, db)}
        rows.append(row)
        deltas.append((da, dap))
        scene_sets.append({sample["a"]["scene"], sample["a_prime"]["scene"]})
        print(json.dumps(row))

    foreign = []
    for i, (source, _) in enumerate(deltas):
        for j, (_, target) in enumerate(deltas):
            # Directional counterparts (X→Y and Y→X) are still the same
            # revisit pair, so they are not valid foreign controls.
            if i != j and not scene_sets[i].intersection(scene_sets[j]):
                foreign.append(cosine(source, target))
    summary = {
        "matched_cosine": sum(row["matched_cosine"] for row in rows) / len(rows),
        "intervening_cosine": sum(row["intervening_cosine"] for row in rows) / len(rows),
        "foreign_cosine": sum(foreign) / len(foreign) if foreign else None,
        "matched_minus_intervening": sum(row["matched_cosine"] - row["intervening_cosine"] for row in rows) / len(rows),
        "matched_minus_foreign": (sum(row["matched_cosine"] for row in rows) / len(rows) - sum(foreign) / len(foreign)) if foreign else None,
    }
    with open(args.out, "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    if args.vectors_out:
        torch.save({"a": torch.stack([pair[0].cpu() for pair in deltas]),
                    "a_prime": torch.stack([pair[1].cpu() for pair in deltas]),
                    "episode_ids": [row["episode"] for row in rows],
                    "scene_sets": [sorted(items) for items in scene_sets]}, args.vectors_out)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
