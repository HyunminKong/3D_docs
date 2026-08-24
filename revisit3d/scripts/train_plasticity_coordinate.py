#!/usr/bin/env python3
"""First-order meta-learn a compact TTT coordinate before adaptation memory.

The decoder is frozen after depth bootstrap.  Only the linear map from the
adapted coordinate z to FiLM modulation is trainable, and it has no effect at
z=0.  A first-order A→B→A' objective directly requires the oracle A update to
beat carried-current, B, and scene-disjoint foreign updates on a held-out A'
query.  This is the go/no-go experiment for reusable *raw* adaptation.
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
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import CompactTTTState, build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def prepare(extractor, tracker, segment, track_side):
    images = segment["context"]["rgb"]
    features = extractor(images)
    prior = tracker(images, query_grid(images.shape[-2], images.shape[-1], track_side, "cuda"))
    return features, prior


def objective(head, prediction, images, prior, smoothness=0.0):
    depth = depth_grid(prediction)
    return track_3d_consistency_loss(depth, prior["intrinsics"], prior["w2c"], prior["track"],
                                     prior["visibility"], prior["confidence"], image_size=images.shape[-2:]) + \
        smoothness * depth_smoothness_loss(depth, images)


def adapt(head, features, prior, images, state, lr, smoothness):
    return head.adapt(features, state,
                      lambda prediction: objective(head, prediction, images, prior, smoothness),
                      # grid_sample has no CUDA double backward; first-order
                      # meta-TTT still learns how fixed online updates should
                      # be decoded into useful geometry corrections.
                      steps=1, learning_rate=lr, create_graph=False, retain_state_gradient=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--track-side", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("plasticity-coordinate training requires CUDA")

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    loaded = torch.load(args.head_checkpoint, map_location="cuda", weights_only=False)
    if loaded.get("head_type", "global") != "global":
        raise SystemExit("this first coordinate experiment is intentionally the global-head control")
    head = build_geometry_head("global", extractor.feature_dim).cuda()
    head.load_state_dict(loaded["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    # Freeze every z=0 pathway.  Only the state-dependent linear map learns.
    coordinate_weight = head.to_scale_shift[-1].weight
    coordinate_weight.requires_grad_(True)
    optimizer = torch.optim.AdamW([coordinate_weight], lr=args.lr, weight_decay=1e-4)

    records = []
    for step, sample in zip(range(args.steps), loader):
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        scenes = {a["scene"], ap["scene"]}
        foreign_item = next(item for item in dataset
                            if not scenes.intersection({item["a"]["scene"], item["a_prime"]["scene"]}))
        foreign_a = to_device(foreign_item["a"], "cuda")
        fa, pa = prepare(extractor, tracker, a, args.track_side)
        fb, pb = prepare(extractor, tracker, b, args.track_side)
        fp, pp = prepare(extractor, tracker, ap, args.track_side)
        ff, pf = prepare(extractor, tracker, foreign_a, args.track_side)
        z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
        za = adapt(head, fa, pa, a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
        zab = adapt(head, fb, pb, b["context"]["rgb"], za, args.ttt_lr, args.smoothness)
        zb = adapt(head, fb, pb, b["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
        zf = adapt(head, ff, pf, foreign_a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
        current = adapt(head, fp, pp, ap["context"]["rgb"], zab, args.ttt_lr, args.smoothness)
        matched = adapt(head, fp, pp, ap["context"]["rgb"],
                        CompactTTTState(zab.value + za.value - z0.value), args.ttt_lr, args.smoothness)
        intervening = adapt(head, fp, pp, ap["context"]["rgb"],
                            CompactTTTState(zab.value + zb.value - z0.value), args.ttt_lr, args.smoothness)
        foreign = adapt(head, fp, pp, ap["context"]["rgb"],
                        CompactTTTState(zab.value + zf.value - z0.value), args.ttt_lr, args.smoothness)
        # The outer query is strictly held out from every inner adaptation.
        query_images = ap["query"]["rgb"]
        fq = extractor(query_images)
        qp = tracker(query_images, query_grid(query_images.shape[-2], query_images.shape[-1], args.track_side, "cuda"))
        query = lambda state: objective(head, head(fq, state), query_images, qp)
        losses = {"current": query(current), "matched": query(matched),
                  "intervening": query(intervening), "foreign": query(foreign)}
        contrast = F.relu(args.margin - (losses["intervening"] - losses["matched"])) + \
            F.relu(args.margin - (losses["foreign"] - losses["matched"]))
        outer = losses["matched"] + F.softplus(losses["matched"] - losses["current"]) + args.contrastive_weight * contrast
        optimizer.zero_grad(set_to_none=True)
        outer.backward()
        torch.nn.utils.clip_grad_norm_([coordinate_weight], 1.0)
        optimizer.step()
        row = {"step": step, "episode": sample["episode_id"], "foreign_episode": foreign_item["episode_id"],
               "outer": float(outer.detach()), "contrast": float(contrast.detach()),
               **{key: float(value.detach()) for key, value in losses.items()}}
        records.append(row); print(json.dumps(row))
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "head_type": "global", "records": records,
                "source_checkpoint": args.head_checkpoint, "protocol": "first_order_oracle_plasticity_coordinate"}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
