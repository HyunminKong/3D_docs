#!/usr/bin/env python3
"""First premise test using frozen-track 3D consistency as the TTT objective."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=0, eps=1e-8))


def adapt_delta(head, extractor, tracker, segment, track_side: int, lr: float, smoothness: float):
    images = segment["context"]["rgb"]
    features = extractor(images)
    prior = tracker(images, query_grid(images.shape[-2], images.shape[-1], track_side, "cuda"))
    initial = head.initial_state(1, device="cuda", dtype=features.dtype)
    def objective(prediction):
        depth = depth_grid(prediction)
        return track_3d_consistency_loss(depth, prior["intrinsics"], prior["w2c"], prior["track"],
                                         prior["visibility"], prior["confidence"], image_size=images.shape[-2:]) + \
            smoothness * depth_smoothness_loss(depth, images)
    adapted, history = head.adapt(features, initial, objective, steps=1, learning_rate=lr)
    return (adapted.value - initial.value).detach().flatten(), float(history[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--track-side", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vectors-out", default="")
    parser.add_argument("--head-type", choices=("global", "slot", "anchored"), default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled probe requires CUDA")
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    loaded = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    head_type = args.head_type or loaded.get("head_type", "global")
    head = build_geometry_head(head_type, extractor.feature_dim).cuda()
    head.load_state_dict(loaded["head"])
    head.eval()
    deltas, scene_sets, rows = [], [], []
    for sample in dataset:
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        with torch.enable_grad():
            da, la = adapt_delta(head, extractor, tracker, a, args.track_side, args.ttt_lr, args.smoothness)
            db, lb = adapt_delta(head, extractor, tracker, b, args.track_side, args.ttt_lr, args.smoothness)
            dap, lap = adapt_delta(head, extractor, tracker, ap, args.track_side, args.ttt_lr, args.smoothness)
        row = {"episode": sample["episode_id"], "loss_a": la, "loss_b": lb, "loss_a_prime": lap,
               "delta_norm_a": float(da.norm()), "delta_norm_b": float(db.norm()), "delta_norm_a_prime": float(dap.norm()),
               "matched_cosine": cosine(da, dap), "intervening_cosine": cosine(da, db)}
        rows.append(row); deltas.append((da, db, dap)); scene_sets.append({a["scene"], ap["scene"]})
        print(json.dumps(row))
    foreign = [cosine(source, target) for i, (source, _, _) in enumerate(deltas)
               for j, (_, _, target) in enumerate(deltas)
               if i != j and not scene_sets[i].intersection(scene_sets[j])]
    summary = {"matched_cosine": sum(row["matched_cosine"] for row in rows) / len(rows),
               "intervening_cosine": sum(row["intervening_cosine"] for row in rows) / len(rows),
               "foreign_cosine": sum(foreign) / len(foreign) if foreign else None,
               "matched_minus_intervening": sum(row["matched_cosine"] - row["intervening_cosine"] for row in rows) / len(rows),
               "matched_minus_foreign": (sum(row["matched_cosine"] for row in rows) / len(rows) - sum(foreign) / len(foreign)) if foreign else None}
    with open(args.out, "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "head_type": head_type, "split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    if args.vectors_out:
        torch.save({"a": torch.stack([entry[0].cpu() for entry in deltas]),
                    "b": torch.stack([entry[1].cpu() for entry in deltas]),
                    "a_prime": torch.stack([entry[2].cpu() for entry in deltas]),
                    "episode_ids": [row["episode"] for row in rows],
                    "scene_sets": [sorted(items) for items in scene_sets]}, args.vectors_out)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
