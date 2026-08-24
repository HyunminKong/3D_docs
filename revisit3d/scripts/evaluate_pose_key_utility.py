#!/usr/bin/env python3
"""Evaluate a 3D map key by utility regret, not arbitrary episode identity."""
from __future__ import annotations

import argparse
import json

import torch

from revisit3d.data import RevisitEpisodeDataset


def centres(segment):
    return torch.linalg.inv(segment["context"]["w2c"])[:, :3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    utility = json.load(open(args.utility))["rows"]
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="val", image_size=(224,224))
    memory = [centres(sample["a"]) for sample in dataset]
    queries = [centres(sample["a_prime"]) for sample in dataset]
    rows=[]
    for query, record in zip(queries, utility):
        key = torch.tensor([torch.cdist(query, item).min() for item in memory])
        values=torch.tensor(record["candidate_utilities"]); oracle=int(values.argmin()); top3=key.topk(3,largest=False).indices
        selected=int(key.argmin()); selected_utility=float(values[selected]); best=float(values[oracle]); top3_best=float(values[top3].min())
        rows.append({"oracle_index":oracle,"pose_top1_index":selected,"oracle_in_pose_top3":bool((top3==oracle).any()),
                     "top1_is_oracle":selected==oracle,"top1_regret":selected_utility-best,
                     "top3_oracle_rerank_regret":top3_best-best,"current_regret":record["current"]-best})
    summary={"pose_top1_oracle_utility":sum(r["top1_is_oracle"] for r in rows)/len(rows),
             "pose_top3_oracle_coverage":sum(r["oracle_in_pose_top3"] for r in rows)/len(rows),
             "mean_top1_regret":sum(r["top1_regret"] for r in rows)/len(rows),
             "mean_top3_oracle_rerank_regret":sum(r["top3_oracle_rerank_regret"] for r in rows)/len(rows),
             "mean_current_regret":sum(r["current_regret"] for r in rows)/len(rows)}
    with open(args.out,"w") as handle: json.dump({"summary":summary,"rows":rows},handle,indent=2)
    print(json.dumps(summary))

if __name__ == "__main__": main()
