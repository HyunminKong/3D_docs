#!/usr/bin/env python3
"""Oracle utility upper bound for a bank of raw local TTT updates.

For every A' query this tests every stored A update by injecting it, performing
one current-context TTT step, then measuring a separate future query segment.
It is intentionally an oracle reranker: retrieval is not involved.  Thus a
failure here rules out key engineering as an explanation for poor recall.
"""
from __future__ import annotations

import argparse
import json

import torch

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import CompactTTTState, build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_learned_local_update import loss, prepare
from revisit3d.scripts.train_oracle_revisit import to_device


def raw_adapt(head, features, prior, images, initial, smoothness):
    return head.adapt(features, initial, lambda pred: loss(pred, images, prior, smoothness),
                      steps=1, learning_rate=1e-2)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="val", image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    checkpoint = torch.load(args.head_checkpoint, map_location="cuda", weights_only=False)
    head = build_geometry_head(checkpoint.get("head_type", "anchored"), extractor.feature_dim).cuda()
    head.load_state_dict(checkpoint["head"])
    head.eval()

    memory, queries = [], []
    with torch.enable_grad():
        for sample in dataset:
            a, ap = (to_device(sample[tag], "cuda") for tag in ("a", "a_prime"))
            fa, pa = prepare(extractor, tracker, a, 8)
            fp, pp = prepare(extractor, tracker, ap, 8)
            qi = ap["query"]["rgb"]
            fq = extractor(qi)
            pq = tracker(qi, query_grid(qi.shape[-2], qi.shape[-1], 8, "cuda"))
            z0 = head.initial_state(1, device="cuda", dtype=fa.dtype)
            za = raw_adapt(head, fa, pa, a["context"]["rgb"], z0, 1e-3)
            memory.append(za.value.detach())
            queries.append((fp, pp, ap["context"]["rgb"], fq, pq, qi, z0))

        rows = []
        for index, (features, prior, images, qfeatures, qprior, qimages, z0) in enumerate(queries):
            utilities = []
            for update in memory:
                injected = CompactTTTState(z0.value + args.scale * update)
                adapted = raw_adapt(head, features, prior, images, injected, 1e-3)
                utilities.append(float(loss(head(qfeatures, adapted), qimages, qprior, 1e-3).detach()))
            utility = torch.tensor(utilities)
            cold = raw_adapt(head, features, prior, images, z0, 1e-3)
            current = float(loss(head(qfeatures, cold), qimages, qprior, 1e-3).detach())
            rank = int((utility.argsort() == index).nonzero()[0]) + 1
            rows.append({"positive_rank": rank, "positive_utility": utilities[index],
                         "best_utility": float(utility.min()), "current_utility": current,
                         "positive_minus_current": utilities[index] - current,
                         "positive_minus_best": utilities[index] - float(utility.min())})
            print(json.dumps(rows[-1]))
    summary = {
        "utility_top1": sum(row["positive_rank"] == 1 for row in rows) / len(rows),
        "utility_recall_at_3": sum(row["positive_rank"] <= 3 for row in rows) / len(rows),
        "mean_positive_rank": sum(row["positive_rank"] for row in rows) / len(rows),
        "mean_positive_minus_current": sum(row["positive_minus_current"] for row in rows) / len(rows),
        "mean_positive_minus_best": sum(row["positive_minus_best"] for row in rows) / len(rows),
    }
    with open(args.out, "w") as handle:
        json.dump({"scale": args.scale, "summary": summary, "rows": rows}, handle, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
