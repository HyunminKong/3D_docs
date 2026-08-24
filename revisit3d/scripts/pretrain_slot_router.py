#!/usr/bin/env python3
"""Self-supervise feature routing with frozen point tracks before slot-state TTT.

Depth bootstrap cannot train a router because the initial adaptable slot values
are zero.  This phase uses only frozen correspondences: corresponding image
locations should select the same slot, while the marginal slot use stays
diverse.  No future/revisit query or geometry pseudo-label is used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import to_device


def sample_assignment(assignment: torch.Tensor, points: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """B,V,P,K token routing probabilities -> B,V,N,K at image-pixel points."""
    batch, views, patches, slots = assignment.shape
    side = int(patches ** 0.5)
    if side * side != patches:
        raise ValueError("slot router expects a square token grid")
    image_h, image_w = image_size
    maps = assignment.permute(0, 1, 3, 2).reshape(batch * views, slots, side, side)
    grid = torch.stack((2 * points[..., 0] / max(image_w - 1, 1) - 1,
                        2 * points[..., 1] / max(image_h - 1, 1) - 1), dim=-1)
    sampled = F.grid_sample(maps, grid.flatten(0, 1).unsqueeze(1), align_corners=True,
                            mode="bilinear", padding_mode="zeros")
    return sampled.squeeze(2).transpose(1, 2).reshape(batch, views, -1, slots)


def router_loss(assignment: torch.Tensor, tracks: dict[str, torch.Tensor], image_size: tuple[int, int]):
    sampled = sample_assignment(assignment, tracks["track"], image_size)
    ref = sampled[:, :1]
    points = tracks["track"]
    height, width = image_size
    valid = ((points[..., 0] >= 0) & (points[..., 0] <= width - 1) &
             (points[..., 1] >= 0) & (points[..., 1] <= height - 1)).to(assignment.dtype)
    weight = (tracks["visibility"] * tracks["confidence"]).to(assignment.dtype)[:, 1:] * valid[:, 1:]
    correspondence = (weight[..., None] * (sampled[:, 1:] - ref).square()).sum() / weight.sum().clamp_min(1e-6)
    probability = assignment.clamp_min(1e-8)
    local_entropy = -(probability * probability.log()).sum(-1).mean()
    marginal = probability.mean(dim=(1, 2)).clamp_min(1e-8)
    marginal_entropy = -(marginal * marginal.log()).sum(-1).mean()
    # Minimizing H(slot|token)-H(slot) maximizes routing mutual information,
    # preventing both uniform and all-in-one-slot collapse.
    mutual_information_loss = local_entropy - marginal_entropy
    return correspondence + 0.1 * mutual_information_loss, {
        "correspondence_loss": correspondence.detach(), "router_mi_loss": mutual_information_loss.detach(),
        "local_entropy": local_entropy.detach(), "marginal_entropy": marginal_entropy.detach(),
        "track_weight": weight.mean().detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="slot depth-bootstrap checkpoint")
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--track-side", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("router pretraining requires CUDA")
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    loaded = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if loaded.get("head_type") not in ("slot", "anchored"):
        raise SystemExit("router pretraining requires a routed-head checkpoint")
    head_type = loaded["head_type"]
    head = build_geometry_head(head_type, extractor.feature_dim).cuda()
    head.load_state_dict(loaded["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    for parameter in list(head.token_router.parameters()) + [head.slot_keys]:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([*head.token_router.parameters(), head.slot_keys], lr=args.lr, weight_decay=1e-4)
    records = []
    for step, sample in zip(range(args.steps), loader):
        segment = to_device(sample["a"], "cuda")
        images = segment["context"]["rgb"]
        features = extractor(images)
        tracks = tracker(images, query_grid(images.shape[-2], images.shape[-1], args.track_side, "cuda"))
        state = head.initial_state(1, device="cuda", dtype=features.dtype)
        loss, stats = router_loss(head(features, state)["slot_assignment"], tracks, images.shape[-2:])
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0); optimizer.step()
        row = {"step": step, "episode": sample["episode_id"], "loss": float(loss.detach()),
               **{key: float(value) for key, value in stats.items()}}
        records.append(row); print(json.dumps(row))
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "head_type": head_type, "records": records,
                "source_checkpoint": args.checkpoint, "bootstrap": "frozen_track_router_consistency"}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
