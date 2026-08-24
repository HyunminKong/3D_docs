#!/usr/bin/env python3
"""Objective-health test for frozen-track 3D consistency TTT."""

from __future__ import annotations

import argparse
import json

import torch

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import build_geometry_head
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def query_grid(height: int, width: int, side: int, device: str) -> torch.Tensor:
    ys = (torch.arange(side, device=device) + 0.5) * height / side
    xs = (torch.arange(side, device=device) + 0.5) * width / side
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--track-side", type=int, default=8)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
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
    rows = []
    for sample in dataset:
        segment = to_device(sample["a"], "cuda")
        images = segment["context"]["rgb"]
        features = extractor(images)
        tracks = tracker(images, query_grid(images.shape[-2], images.shape[-1], args.track_side, "cuda"))
        state = head.initial_state(1, device="cuda", dtype=features.dtype)
        state.value.requires_grad_(True)
        prediction = head(features, state)
        depth = depth_grid(prediction)
        loss, stats = track_3d_consistency_loss(depth, tracks["intrinsics"], tracks["w2c"], tracks["track"],
                                                 tracks["visibility"], tracks["confidence"],
                                                 image_size=images.shape[-2:], return_stats=True)
        loss = loss + args.smoothness * depth_smoothness_loss(depth, images)
        gradient, = torch.autograd.grad(loss, state.value)
        row = {"episode": sample["episode_id"], "track_3d_loss": float(loss.detach()),
               "state_grad_norm": float(gradient.norm().detach()), "depth_mean": float(depth.mean().detach()),
               "depth_std": float(depth.std().detach()), **{key: float(value.detach()) for key, value in stats.items()}}
        rows.append(row); print(json.dumps(row))
    keys = [key for key in rows[0] if key != "episode"]
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    with open(args.out, "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "head_type": head_type, "split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
