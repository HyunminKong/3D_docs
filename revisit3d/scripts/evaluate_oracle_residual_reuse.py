#!/usr/bin/env python3
"""Causal oracle test: does a matched residual update beat a foreign one?

This is still not a memory framework.  The earlier traversal is supplied by
the manifest oracle, and the residual is a fixed global-subspace subtraction
estimated only from the train split.  The held-out A' query is never used by
online adaptation.
"""

from __future__ import annotations

import argparse
import json

import torch

from revisit3d.backbones import FrozenVGGTDepthTeacher, FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, reprojection_loss
from revisit3d.models import CompactTTTState, StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import depth_grid, to_device


def prepare(extractor, teacher, segment):
    features = extractor(segment["context"]["rgb"])
    side = int(features.shape[2] ** 0.5)
    return features, teacher(segment["context"]["rgb"], (side, side))


def loss_for(prediction, images, geometry, smoothness):
    depth = depth_grid(prediction)
    return reprojection_loss(depth, images, geometry["intrinsics"], geometry["w2c"]) + \
        smoothness * depth_smoothness_loss(depth, images)


def adapt(head, features, geometry, images, initial, lr, smoothness):
    return head.adapt(features, initial,
                      lambda prediction: loss_for(prediction, images, geometry, smoothness),
                      steps=1, learning_rate=lr)[0]


def query_loss(head, extractor, teacher, segment, state):
    features = extractor(segment["query"]["rgb"])
    side = int(features.shape[2] ** 0.5)
    geometry = teacher(segment["query"]["rgb"], (side, side))
    return loss_for(head(features, state), segment["query"]["rgb"], geometry, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-vectors", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--remove-rank", type=int, default=1)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled probe requires CUDA")

    vectors = torch.load(args.train_vectors, map_location="cuda", weights_only=False)
    train_updates = torch.cat((vectors["a"], vectors["a_prime"]), dim=0).float().cuda()
    mean = train_updates.mean(0, keepdim=True)
    _, _, basis = torch.linalg.svd(train_updates - mean, full_matrices=False)
    basis = basis[:args.remove_rank]
    def residual(update):
        centred = update - mean
        return centred - (centred @ basis.T) @ basis if args.remove_rank else centred

    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    train_dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    teacher = FrozenVGGTDepthTeacher(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).cuda()
    head.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=False)["head"])
    head.eval()

    rows = []
    for sample in dataset:
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        current_scenes = {a["scene"], ap["scene"]}
        foreign_sample = next(item for item in train_dataset
                              if not current_scenes.intersection({item["a"]["scene"], item["a_prime"]["scene"]}))
        foreign_a = to_device(foreign_sample["a"], "cuda")
        with torch.enable_grad():
            fa, ga = prepare(extractor, teacher, a)
            fb, gb = prepare(extractor, teacher, b)
            fap, gap = prepare(extractor, teacher, ap)
            z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
            za = adapt(head, fa, ga, a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            zab = adapt(head, fb, gb, b["context"]["rgb"], za, args.ttt_lr, args.smoothness)
            cold = adapt(head, fap, gap, ap["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            current = adapt(head, fap, gap, ap["context"]["rgb"], zab, args.ttt_lr, args.smoothness)
            matched = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + residual(za.value - z0.value)), args.ttt_lr, args.smoothness)
            ff, fg = prepare(extractor, teacher, foreign_a)
            zf = adapt(head, ff, fg, foreign_a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            foreign = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + residual(zf.value - z0.value)), args.ttt_lr, args.smoothness)
            losses = {"cold": query_loss(head, extractor, teacher, ap, cold),
                      "current": query_loss(head, extractor, teacher, ap, current),
                      "matched": query_loss(head, extractor, teacher, ap, matched),
                      "foreign": query_loss(head, extractor, teacher, ap, foreign)}
        row = {"episode": sample["episode_id"], "foreign_episode": foreign_sample["episode_id"],
               **{key: float(value.detach()) for key, value in losses.items()}}
        row["matched_minus_current"] = row["matched"] - row["current"]
        row["matched_minus_foreign"] = row["matched"] - row["foreign"]
        rows.append(row)
        print(json.dumps(row))
    keys = ("cold", "current", "matched", "foreign", "matched_minus_current", "matched_minus_foreign")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    output = {"checkpoint": args.checkpoint, "split": args.split, "remove_rank": args.remove_rank,
              "rows": rows, "summary": summary}
    with open(args.out, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
