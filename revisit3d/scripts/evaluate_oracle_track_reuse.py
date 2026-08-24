#!/usr/bin/env python3
"""Causal oracle reuse test under frozen-track 3D consistency TTT."""

from __future__ import annotations

import argparse
import json

import torch

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import CompactTTTState, build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def prepare(extractor, tracker, segment, track_side):
    images = segment["context"]["rgb"]
    return extractor(images), tracker(images, query_grid(images.shape[-2], images.shape[-1], track_side, "cuda"))


def objective(prediction, images, prior, smoothness=0.0):
    depth = depth_grid(prediction)
    return track_3d_consistency_loss(depth, prior["intrinsics"], prior["w2c"], prior["track"],
                                     prior["visibility"], prior["confidence"], image_size=images.shape[-2:]) + \
        smoothness * depth_smoothness_loss(depth, images)


def adapt(head, features, prior, images, state, lr, smoothness):
    return head.adapt(features, state, lambda pred: objective(pred, images, prior, smoothness),
                      steps=1, learning_rate=lr)[0]


def query_loss(head, extractor, tracker, segment, state, track_side):
    images = segment["query"]["rgb"]
    features = extractor(images)
    prior = tracker(images, query_grid(images.shape[-2], images.shape[-1], track_side, "cuda"))
    return objective(head(features, state), images, prior)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--track-side", type=int, default=8)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--head-type", choices=("global", "slot", "anchored"), default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled probe requires CUDA")
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    train_dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    loaded = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    head_type = args.head_type or loaded.get("head_type", "global")
    head = build_geometry_head(head_type, extractor.feature_dim).cuda()
    head.load_state_dict(loaded["head"])
    head.eval()
    generator = torch.Generator(device="cuda").manual_seed(args.random_seed)
    rows = []
    for sample in dataset:
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        scenes = {a["scene"], ap["scene"]}
        foreign_item = next(item for item in train_dataset
                            if not scenes.intersection({item["a"]["scene"], item["a_prime"]["scene"]}))
        foreign_a = to_device(foreign_item["a"], "cuda")
        with torch.enable_grad():
            fa, pa = prepare(extractor, tracker, a, args.track_side)
            fb, pb = prepare(extractor, tracker, b, args.track_side)
            fp, pp = prepare(extractor, tracker, ap, args.track_side)
            ff, pf = prepare(extractor, tracker, foreign_a, args.track_side)
            z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
            za = adapt(head, fa, pa, a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            zab = adapt(head, fb, pb, b["context"]["rgb"], za, args.ttt_lr, args.smoothness)
            zb = adapt(head, fb, pb, b["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            zf = adapt(head, ff, pf, foreign_a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            cold = adapt(head, fp, pp, ap["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            current = adapt(head, fp, pp, ap["context"]["rgb"], zab, args.ttt_lr, args.smoothness)
            matched = adapt(head, fp, pp, ap["context"]["rgb"],
                            CompactTTTState(zab.value + za.value - z0.value), args.ttt_lr, args.smoothness)
            intervening = adapt(head, fp, pp, ap["context"]["rgb"],
                                CompactTTTState(zab.value + zb.value - z0.value), args.ttt_lr, args.smoothness)
            foreign = adapt(head, fp, pp, ap["context"]["rgb"],
                            CompactTTTState(zab.value + zf.value - z0.value), args.ttt_lr, args.smoothness)
            update_norm = (za.value - z0.value).norm(dim=-1, keepdim=True)
            random_update = torch.randn(za.value.shape, device="cuda", dtype=za.value.dtype, generator=generator)
            random_update = random_update / random_update.norm(dim=-1, keepdim=True).clamp_min(1e-8) * update_norm
            random_state = adapt(head, fp, pp, ap["context"]["rgb"],
                                 CompactTTTState(zab.value + random_update), args.ttt_lr, args.smoothness)
            values = {"cold": query_loss(head, extractor, tracker, ap, cold, args.track_side),
                      "current": query_loss(head, extractor, tracker, ap, current, args.track_side),
                      "matched": query_loss(head, extractor, tracker, ap, matched, args.track_side),
                      "intervening": query_loss(head, extractor, tracker, ap, intervening, args.track_side),
                      "foreign": query_loss(head, extractor, tracker, ap, foreign, args.track_side),
                      "random": query_loss(head, extractor, tracker, ap, random_state, args.track_side)}
        row = {"episode": sample["episode_id"], "foreign_episode": foreign_item["episode_id"],
               **{key: float(value.detach()) for key, value in values.items()}}
        row["matched_minus_current"] = row["matched"] - row["current"]
        row["matched_minus_intervening"] = row["matched"] - row["intervening"]
        row["matched_minus_foreign"] = row["matched"] - row["foreign"]
        row["matched_minus_random"] = row["matched"] - row["random"]
        rows.append(row); print(json.dumps(row))
    keys = ("cold", "current", "matched", "intervening", "foreign", "random", "matched_minus_current",
            "matched_minus_intervening", "matched_minus_foreign", "matched_minus_random")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    with open(args.out, "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "head_type": head_type, "split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
