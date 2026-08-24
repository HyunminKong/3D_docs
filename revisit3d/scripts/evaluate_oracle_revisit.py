#!/usr/bin/env python3
"""Evaluate cold/current/oracle-matched/foreign compact-state transfer.

This evaluator is intentionally the last step before retrieval.  It supplies
the correct earlier episode as an oracle, then asks whether it improves held-out
A' reprojection over both a continued current state and an unrelated episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import relative_w2c_from_twist, reprojection_loss
from revisit3d.models import CompactTTTState, SignedResidualTransport, StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import depth_grid, segment_loss, to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pose-source", choices=("known", "predicted"), default="predicted")
    parser.add_argument("--ttt-steps", type=int, default=1)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--out", default="revisit3d/results/oracle_eval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    foreign_data = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").to(device)
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).to(device)
    transport = SignedResidualTransport(extractor.feature_dim, state_dim=32, hidden_dim=128).to(device)
    head.load_state_dict(saved["head"]); transport.load_state_dict(saved["transport"])
    head.eval(); transport.eval()

    def feature(segment, key="context"):
        return extractor(segment[key]["rgb"])

    def loss_for(prediction, segment, key="context"):
        if key == "context":
            return segment_loss(prediction, segment, smoothness=args.smoothness, pose_source=args.pose_source)
        camera = segment[key]
        transform = camera["w2c"] if args.pose_source == "known" else relative_w2c_from_twist(prediction["relative_pose"])
        return reprojection_loss(depth_grid(prediction), camera["rgb"], camera["intrinsics"], transform)

    @torch.enable_grad()
    def adapt(features, state, segment, tag):
        return head.adapt(features, state, lambda pred: loss_for(pred, segment),
                          steps=args.ttt_steps, learning_rate=args.ttt_lr)[0]

    rows = []
    for index in range(len(data)):
        sample = {name: to_device(data[index][name], device) for name in ("a", "b", "a_prime")}
        foreign_sample = to_device(foreign_data[index % len(foreign_data)]["a"], device)
        fa, fb, fap, fquery = (feature(sample["a"]), feature(sample["b"]),
                                feature(sample["a_prime"]), feature(sample["a_prime"], "query"))
        fforeign = feature(foreign_sample)
        initial = head.initial_state(1, device=device, dtype=fa.dtype)
        state_a = adapt(fa, initial, sample["a"], "a")
        state_ab = adapt(fb, state_a, sample["b"], "b")
        foreign_state = adapt(fforeign, initial, foreign_sample, "foreign")
        cold = adapt(fap, initial, sample["a_prime"], "a_prime")
        current = adapt(fap, state_ab, sample["a_prime"], "a_prime")
        matched = adapt(fap, CompactTTTState(state_ab.value + transport(state_a, fap).value), sample["a_prime"], "a_prime")
        foreign = adapt(fap, CompactTTTState(state_ab.value + transport(foreign_state, fap).value), sample["a_prime"], "a_prime")
        with torch.no_grad():
            score = lambda state: float(loss_for(head(fquery, state), sample["a_prime"], key="query"))
            row = {"episode": data.records[index]["episode_id"], "cold": score(cold), "current": score(current),
                   "matched": score(matched), "foreign": score(foreign)}
            row["matched_minus_current"] = row["matched"] - row["current"]
            row["matched_minus_foreign"] = row["matched"] - row["foreign"]
            rows.append(row)
            print(json.dumps(row))

    summary = {key: float(sum(row[key] for row in rows) / len(rows))
               for key in ("cold", "current", "matched", "foreign", "matched_minus_current", "matched_minus_foreign")}
    payload = {"checkpoint": args.checkpoint, "pose_source": args.pose_source, "rows": rows, "summary": summary}
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"summary": summary, "out": str(path)}))


if __name__ == "__main__":
    main()
