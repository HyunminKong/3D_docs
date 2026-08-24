#!/usr/bin/env python3
"""Can an unmodified local TTT gradient disambiguate revisit candidates?

This is a pre-framework diagnostic.  It deliberately uses no learned update
rule: if raw gradients cannot identify a revisit within key-retrieved
candidates, an update-memory routing mechanism has no signal to exploit.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import build_geometry_head
from revisit3d.models.local_key import LocalTokenKey
from revisit3d.scripts.diagnose_track_update_alignment import adapt_delta
from revisit3d.scripts.train_oracle_revisit import to_device


def key_tokens(extractor, key, segment):
    features = extractor(segment["context"]["rgb"])[0]
    views = torch.linspace(0, features.shape[0] - 1, 4, device=features.device).long()
    return key(features[views, ::4].reshape(-1, features.shape[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--key-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="val", image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    checkpoint = torch.load(args.head_checkpoint, map_location="cuda", weights_only=False)
    head = build_geometry_head(checkpoint.get("head_type", "anchored"), extractor.feature_dim).cuda()
    head.load_state_dict(checkpoint["head"])
    head.eval()
    key = LocalTokenKey(extractor.feature_dim).cuda()
    key.load_state_dict(torch.load(args.key_checkpoint, map_location="cuda", weights_only=False)["key"])
    key.eval()

    memory, queries = [], []
    with torch.enable_grad():
        for sample in dataset:
            a, ap = (to_device(sample[tag], "cuda") for tag in ("a", "a_prime"))
            da, _ = adapt_delta(head, extractor, tracker, a, 8, 1e-2, 1e-3)
            dap, _ = adapt_delta(head, extractor, tracker, ap, 8, 1e-2, 1e-3)
            memory.append((key_tokens(extractor, key, a).detach(), da))
            queries.append((key_tokens(extractor, key, ap).detach(), dap))

    rows = []
    for index, (query_key, query_update) in enumerate(queries):
        key_scores = torch.tensor([float(key.score(query_key, memory_key)) for memory_key, _ in memory])
        compat = torch.tensor([float(F.cosine_similarity(query_update, memory_update, dim=0))
                               for _, memory_update in memory])
        candidates = key_scores.topk(args.k).indices
        choice = int(candidates[compat[candidates].argmax()])
        key_rank = int((key_scores.argsort(descending=True) == index).nonzero()[0]) + 1
        compat_rank = int((compat.argsort(descending=True) == index).nonzero()[0]) + 1
        rows.append({"key_rank": key_rank, "compat_rank": compat_rank,
                     "positive_compat": float(compat[index]),
                     "mean_negative_compat": float((compat.sum() - compat[index]) / (len(compat) - 1)),
                     "rerank_correct": choice == index})

    updates = torch.stack([update.float().cpu() for _, update in memory])
    singular = torch.linalg.svdvals(updates - updates.mean(0, keepdim=True))
    energy = singular.square() / singular.square().sum()
    summary = {
        "key_recall_at_k": sum(row["key_rank"] <= args.k for row in rows) / len(rows),
        "rerank_top1": sum(row["rerank_correct"] for row in rows) / len(rows),
        "compat_top1": sum(row["compat_rank"] == 1 for row in rows) / len(rows),
        "mean_positive_compat": sum(row["positive_compat"] for row in rows) / len(rows),
        "mean_negative_compat": sum(row["mean_negative_compat"] for row in rows) / len(rows),
        "update_centered_energy_top1": float(energy[0]),
        "update_centered_energy_top3": float(energy[:3].sum()),
    }
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
