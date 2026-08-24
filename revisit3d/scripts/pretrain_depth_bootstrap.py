#!/usr/bin/env python3
"""Bootstrap only the new depth decoder before testing compact-state TTT.

The target is a *frozen foundation pseudo-label*, never a test-time input and
never a final reconstruction score.  This is a calibration control: the prior
experiment showed that an untrained head predicts flat, out-of-view depth, so
the online reprojection objective has no identifiable state update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from revisit3d.backbones import FrozenVGGTDepthTeacher, FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.scripts.train_oracle_revisit import to_device
from revisit3d.models import build_geometry_head


def depth_bootstrap_loss(prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> torch.Tensor:
    predicted = prediction["depth"].squeeze(-1)
    side = int(predicted.shape[-1] ** 0.5)
    if side * side != predicted.shape[-1]:
        raise ValueError("depth bootstrap expects a square token grid")
    predicted = predicted.reshape(*predicted.shape[:2], side, side).clamp_min(1e-4)
    teacher = target["depth"].clamp_min(1e-4)
    # Log depth preserves the foundation's arbitrary global scale while still
    # forcing non-flat local geometry.  Confidence is a fixed reliability
    # weight, normalized to avoid a changing loss scale.
    weight = target["confidence"].detach().clamp_min(1e-3)
    return (weight * F.smooth_l1_loss(predicted.log(), teacher.log(), reduction="none")).sum() / weight.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="revisit3d/checkpoints/depth_bootstrap.pt")
    parser.add_argument("--head-type", choices=("global", "slot", "anchored"), default="global")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This bootstrap requires CUDA")

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    teacher = FrozenVGGTDepthTeacher(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    head = build_geometry_head(args.head_type, extractor.feature_dim).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    records = []
    for step, sample in zip(range(args.steps), loader):
        # Use just A here: this is decoder calibration, not revisit training.
        segment = to_device(sample["a"], "cuda")
        images = segment["context"]["rgb"]
        features = extractor(images)
        grid = int(features.shape[2] ** 0.5)
        target = teacher(images, (grid, grid))
        state = head.initial_state(1, device="cuda", dtype=features.dtype)
        loss = depth_bootstrap_loss(head(features, state), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        record = {"step": step, "episode": sample["episode_id"], "depth_bootstrap_loss": float(loss.detach())}
        records.append(record)
        print(json.dumps(record))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "records": records,
                "bootstrap": "frozen_vggt_depth_pseudo_label", "head_type": args.head_type}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
