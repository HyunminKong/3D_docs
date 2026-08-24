#!/usr/bin/env python3
"""Train the memory-free oracle transport against held-out revisit utility.

The custom geometry head is frozen after depth bootstrap.  The only trainable
object is the small transport from an oracle A state and current A' context to
an initial compact-state correction.  This prevents a decoder-wide global
improvement from masquerading as reusable adaptation.
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
from revisit3d.losses import depth_smoothness_loss, reprojection_loss
from revisit3d.models import CompactTTTState, SignedResidualTransport, StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def prepare(extractor, teacher, segment):
    features = extractor(segment["context"]["rgb"])
    side = int(features.shape[2] ** 0.5)
    return features, teacher(segment["context"]["rgb"], (side, side))


def objective(prediction, images, geometry, smoothness=0.0):
    depth = depth_grid(prediction)
    return reprojection_loss(depth, images, geometry["intrinsics"], geometry["w2c"]) + \
        smoothness * depth_smoothness_loss(depth, images)


def adapt(head, features, geometry, images, initial, lr, smoothness, retain_state_gradient=False):
    return head.adapt(features, initial,
                      lambda prediction: objective(prediction, images, geometry, smoothness),
                      steps=1, learning_rate=lr, retain_state_gradient=retain_state_gradient)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=5e-3)
    parser.add_argument("--out", default="revisit3d/checkpoints/oracle_utility_transport.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled oracle training requires CUDA")

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    teacher = FrozenVGGTDepthTeacher(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).cuda()
    head.load_state_dict(torch.load(args.head_checkpoint, map_location="cuda", weights_only=False)["head"])
    head.eval().requires_grad_(False)
    transport = SignedResidualTransport(extractor.feature_dim, state_dim=32, hidden_dim=128).cuda()
    optimizer = torch.optim.AdamW(transport.parameters(), lr=args.lr, weight_decay=1e-4)

    records = []
    for step, sample in zip(range(args.steps), loader):
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        current_scenes = {a["scene"], ap["scene"]}
        foreign_item = next(item for item in dataset
                            if not current_scenes.intersection({item["a"]["scene"], item["a_prime"]["scene"]}))
        foreign_a = to_device(foreign_item["a"], "cuda")
        fa, ga = prepare(extractor, teacher, a)
        fb, gb = prepare(extractor, teacher, b)
        fap, gap = prepare(extractor, teacher, ap)
        ff, fg = prepare(extractor, teacher, foreign_a)
        z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
        with torch.enable_grad():
            za = adapt(head, fa, ga, a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            zab = adapt(head, fb, gb, b["context"]["rgb"], za, args.ttt_lr, args.smoothness)
            zf = adapt(head, ff, fg, foreign_a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            cold = adapt(head, fap, gap, ap["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            current = adapt(head, fap, gap, ap["context"]["rgb"], zab, args.ttt_lr, args.smoothness)
            matched = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + transport(za, fap).value), args.ttt_lr, args.smoothness,
                            retain_state_gradient=True)
            foreign = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + transport(zf, fap).value), args.ttt_lr, args.smoothness,
                            retain_state_gradient=True)
            fq = extractor(ap["query"]["rgb"])
            side = int(fq.shape[2] ** 0.5)
            gq = teacher(ap["query"]["rgb"], (side, side))
            query = lambda state: objective(head(fq, state), ap["query"]["rgb"], gq)
            losses = {"cold": query(cold), "current": query(current),
                      "matched": query(matched), "foreign": query(foreign)}
            loss = losses["matched"] + F.softplus(losses["matched"] - losses["current"])
            loss = loss + args.contrastive_weight * F.relu(
                args.contrastive_margin - (losses["foreign"] - losses["matched"])
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transport.parameters(), 1.0)
        optimizer.step()
        record = {"step": step, "episode": sample["episode_id"], "foreign_episode": foreign_item["episode_id"],
                  "objective": float(loss.detach()), **{key: float(value.detach()) for key, value in losses.items()}}
        records.append(record)
        print(json.dumps(record))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "transport": transport.state_dict(), "records": records,
                "head_checkpoint": args.head_checkpoint, "protocol": "frozen_foundation_camera_oracle_transport"}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
