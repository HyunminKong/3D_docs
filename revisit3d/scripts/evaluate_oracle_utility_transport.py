#!/usr/bin/env python3
"""Held-out evaluation for the frozen-head oracle utility transport."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTDepthTeacher, FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import CompactTTTState, SignedResidualTransport, StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import to_device
from revisit3d.scripts.train_oracle_utility_transport import adapt, objective, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This controlled oracle evaluation requires CUDA")
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    train_dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    teacher = FrozenVGGTDepthTeacher(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    loaded = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).cuda()
    head.load_state_dict(loaded["head"])
    head.eval().requires_grad_(False)
    transport = SignedResidualTransport(extractor.feature_dim, state_dim=32, hidden_dim=128).cuda()
    transport.load_state_dict(loaded["transport"])
    transport.eval()
    rows = []
    for sample in dataset:
        a, b, ap = (to_device(sample[tag], "cuda") for tag in ("a", "b", "a_prime"))
        scenes = {a["scene"], ap["scene"]}
        foreign_item = next(item for item in train_dataset
                            if not scenes.intersection({item["a"]["scene"], item["a_prime"]["scene"]}))
        foreign_a = to_device(foreign_item["a"], "cuda")
        with torch.enable_grad():
            fa, ga = prepare(extractor, teacher, a); fb, gb = prepare(extractor, teacher, b)
            fap, gap = prepare(extractor, teacher, ap); ff, fg = prepare(extractor, teacher, foreign_a)
            z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
            za = adapt(head, fa, ga, a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            zab = adapt(head, fb, gb, b["context"]["rgb"], za, args.ttt_lr, args.smoothness)
            zf = adapt(head, ff, fg, foreign_a["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            cold = adapt(head, fap, gap, ap["context"]["rgb"], z0, args.ttt_lr, args.smoothness)
            current = adapt(head, fap, gap, ap["context"]["rgb"], zab, args.ttt_lr, args.smoothness)
            matched_prior, foreign_prior = transport(za, fap).value, transport(zf, fap).value
            matched = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + matched_prior), args.ttt_lr, args.smoothness)
            foreign = adapt(head, fap, gap, ap["context"]["rgb"],
                            CompactTTTState(zab.value + foreign_prior), args.ttt_lr, args.smoothness)
            fq = extractor(ap["query"]["rgb"]); side = int(fq.shape[2] ** 0.5)
            gq = teacher(ap["query"]["rgb"], (side, side))
            query = lambda state: objective(head(fq, state), ap["query"]["rgb"], gq)
            values = {"cold": query(cold), "current": query(current), "matched": query(matched), "foreign": query(foreign)}
        row = {"episode": sample["episode_id"], "foreign_episode": foreign_item["episode_id"],
               **{key: float(value.detach()) for key, value in values.items()}}
        row["matched_minus_current"] = row["matched"] - row["current"]
        row["matched_minus_foreign"] = row["matched"] - row["foreign"]
        row["matched_prior_norm"] = float(matched_prior.norm().detach())
        row["foreign_prior_norm"] = float(foreign_prior.norm().detach())
        row["prior_difference_norm"] = float((matched_prior - foreign_prior).norm().detach())
        row["prior_cosine"] = float(F.cosine_similarity(matched_prior, foreign_prior, dim=-1).mean().detach())
        rows.append(row); print(json.dumps(row))
    keys = ("cold", "current", "matched", "foreign", "matched_minus_current", "matched_minus_foreign",
            "matched_prior_norm", "foreign_prior_norm", "prior_difference_norm", "prior_cosine")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    with open(args.out, "w") as handle:
        json.dump({"checkpoint": args.checkpoint, "split": args.split, "rows": rows, "summary": summary}, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
