#!/usr/bin/env python3
"""Oracle world-coordinate version of the dense plasticity-transport probe.

Known camera poses are used *only* to establish an upper bound: if updates are
not selective even when addressed by a shared 3D coordinate system, no online
pose/map design can rescue the present update objective.  A positive gap says
that future work must estimate and maintain that coordinate system online.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import build_geometry_head
from revisit3d.scripts.evaluate_dense_transport_probe import adapt, dense_depth, objective
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_learned_local_update import prepare
from revisit3d.scripts.train_oracle_revisit import to_device


def world_points(head, features, camera):
    depth = dense_depth(head, features, torch.zeros(features.shape[:3], device=features.device, dtype=features.dtype))
    batch, views, height, width = depth.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                            torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij")
    # token-centre pixels in the resized input image coordinate system
    u = (xx.reshape(1, 1, -1) + .5) * (224 / width)
    v = (yy.reshape(1, 1, -1) + .5) * (224 / height)
    fx, fy, cx, cy = camera["intrinsics"].to(depth.dtype).unbind(-1)
    z = depth.reshape(batch, views, -1)
    xyz = torch.stack(((u - cx[..., None]) / fx[..., None] * z,
                       (v - cy[..., None]) / fy[..., None] * z, z,
                       torch.ones_like(z)), dim=-1)
    c2w = torch.linalg.inv(camera["w2c"].to(depth.dtype))
    return torch.einsum("bvij,bvnj->bvni", c2w, xyz)[..., :3]


def transport(source_points, source_atom, target_points, radius, source_features=None, target_features=None,
              appearance_weight=0.0):
    distance = torch.cdist(target_points.flatten(1, 2), source_points.flatten(1, 2))
    logits = -distance / radius
    if appearance_weight:
        source = F.normalize(source_features.flatten(1, 2), dim=-1)
        target = F.normalize(target_features.flatten(1, 2), dim=-1)
        logits = logits + appearance_weight * (target @ source.transpose(-1, -2))
    weights = torch.softmax(logits, dim=-1)
    values = source_atom.flatten(1)
    return (weights @ values.unsqueeze(-1)).squeeze(-1).reshape(*target_points.shape[:3])


def future_loss(head, source_points, source_atom, source_features, query_features, query_camera, query_prior, query_images,
                radius, smoothness, appearance_weight):
    query_points = world_points(head, query_features, query_camera)
    query_atom = transport(source_points, source_atom, query_points, radius, source_features, query_features, appearance_weight)
    return float(objective(head, query_features, query_atom, query_prior, query_images, smoothness).detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--appearance-weight", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    tracker = FrozenVGGTGeometryTracker(args.vggt_checkpoint, repo_root="FastVGGT").cuda()
    checkpoint = torch.load(args.head_checkpoint, map_location="cuda", weights_only=False)
    head = build_geometry_head(checkpoint.get("head_type", "anchored"), extractor.feature_dim).cuda()
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    cached = []
    with torch.enable_grad():
        for sample in dataset:
            a, ap = (to_device(sample[tag], "cuda") for tag in ("a", "a_prime"))
            fa, pa = prepare(extractor, tracker, a, 8)
            fp, pp = prepare(extractor, tracker, ap, 8)
            qi = ap["query"]["rgb"]
            fq = extractor(qi)
            pq = tracker(qi, query_grid(qi.shape[-2], qi.shape[-1], 8, "cuda"))
            zero = torch.zeros(fa.shape[:3], device="cuda", dtype=fa.dtype)
            atom_a, _ = adapt(head, fa, pa, a["context"]["rgb"], zero, args.step_size, 1e-3)
            cached.append((fa.detach(), world_points(head, fa, a["context"]), atom_a, fp.detach(), pp, ap["context"]["rgb"],
                           world_points(head, fp, ap["context"]), fq.detach(), ap["query"], pq, qi))
        rows = []
        for index, (_, _, _, current_features, current_prior, current_images, current_points, query_features, query_camera, query_prior, query_images) in enumerate(cached):
            base = torch.zeros(current_features.shape[:3], device="cuda", dtype=current_features.dtype)
            current_atom, _ = adapt(head, current_features, current_prior, current_images, base, args.step_size, 1e-3)
            current_context_objective = float(objective(head, current_features, current_atom, current_prior,
                                                        current_images, 1e-3).detach())
            current = future_loss(head, current_points, current_atom, current_features, query_features, query_camera, query_prior, query_images,
                                  args.radius, 1e-3, args.appearance_weight)
            candidates = []
            candidate_current_objectives = []
            for source_features, source_points, source_atom, _, _, _, _, _, _, _, _ in cached:
                carried = transport(source_points, source_atom, current_points, args.radius, source_features, current_features,
                                    args.appearance_weight)
                adapted, _ = adapt(head, current_features, current_prior, current_images, carried, args.step_size, 1e-3)
                candidate_current_objectives.append(float(objective(head, current_features, adapted, current_prior,
                                                                  current_images, 1e-3).detach()))
                candidates.append(future_loss(head, current_points, adapted, current_features, query_features, query_camera, query_prior,
                                              query_images, args.radius, 1e-3, args.appearance_weight))
            values = torch.tensor(candidates); positive = candidates[index]; foreign = [v for j,v in enumerate(candidates) if j != index]
            rank = int((values.argsort() == index).nonzero()[0]) + 1
            rows.append({"positive_rank": rank, "current": current, "current_context_objective": current_context_objective, "matched": positive,
                         "mean_foreign":sum(foreign)/len(foreign), "matched_minus_current":positive-current,
                         "matched_minus_foreign":positive-sum(foreign)/len(foreign),
                         "candidate_utilities": candidates,
                         "candidate_current_objectives": candidate_current_objectives})
            print(json.dumps(rows[-1]))
    summary={"utility_top1":sum(r["positive_rank"]==1 for r in rows)/len(rows),
             "utility_recall_at_3":sum(r["positive_rank"]<=3 for r in rows)/len(rows),
             "mean_positive_rank":sum(r["positive_rank"] for r in rows)/len(rows),
             "matched_minus_current":sum(r["matched_minus_current"] for r in rows)/len(rows),
             "matched_minus_foreign":sum(r["matched_minus_foreign"] for r in rows)/len(rows)}
    with open(args.out,"w") as handle: json.dump({"summary":summary,"rows":rows,"radius":args.radius,
                                                    "appearance_weight":args.appearance_weight},handle,indent=2)
    print(json.dumps(summary))


if __name__ == "__main__": main()
