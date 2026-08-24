#!/usr/bin/env python3
"""Oracle A→A' reuse probe with correspondence-transported dense TTT atoms.

Unlike a global/slot state, an atom here is one scalar residual per frozen
feature token.  It is transported to a new view by frozen-feature attention
before local TTT and again before evaluating future frames.  This isolates the
minimal property missing from the collapsed-vector experiments: a persistent,
spatially addressable update object.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, track_3d_consistency_loss
from revisit3d.models import build_geometry_head
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_learned_local_update import prepare
from revisit3d.scripts.train_oracle_revisit import to_device


def flat_features(features):
    return F.normalize(features.flatten(1, 2), dim=-1)


def transport(source_features, source_atom, target_features, temperature):
    """Move a scalar atom between token sets using frozen local correspondence."""
    source = flat_features(source_features)
    target = flat_features(target_features)
    logits = target @ source.transpose(-1, -2) / temperature
    weights = torch.softmax(logits, dim=-1)
    values = source_atom.flatten(1)
    return (weights @ values.unsqueeze(-1)).squeeze(-1).reshape(*target_features.shape[:3])


def dense_depth(head, features, atom):
    zero = head.initial_state(features.shape[0], device=features.device, dtype=features.dtype)
    base = head(features, zero)["depth"].squeeze(-1)
    side = int(base.shape[-1] ** 0.5)
    if side * side != base.shape[-1]:
        raise ValueError("feature token grid must be square")
    return (base * atom.clamp(-0.5, 0.5).exp()).reshape(base.shape[0], base.shape[1], side, side)


def objective(head, features, atom, prior, images, smoothness):
    depth = dense_depth(head, features, atom)
    return track_3d_consistency_loss(depth, prior["intrinsics"], prior["w2c"], prior["track"],
                                     prior["visibility"], prior["confidence"], image_size=images.shape[-2:]) + \
        smoothness * depth_smoothness_loss(depth, images)


def adapt(head, features, prior, images, initial, step_size, smoothness):
    atom = initial.detach().requires_grad_(True)
    inner = objective(head, features, atom, prior, images, smoothness)
    gradient, = torch.autograd.grad(inner, atom)
    # Per-segment normalization prevents the storage object from encoding only
    # an arbitrary gradient magnitude; its local pattern remains untouched.
    normalized = gradient / gradient.abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (atom - step_size * normalized).detach(), float(inner.detach())


def future_loss(head, source_features, source_atom, query_features, query_prior, query_images, temperature, smoothness):
    query_atom = transport(source_features, source_atom, query_features, temperature)
    return float(objective(head, query_features, query_atom, query_prior, query_images, smoothness).detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="val", image_size=(224, 224))
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
            zero_a = torch.zeros(fa.shape[:3], device="cuda", dtype=fa.dtype)
            delta_a, _ = adapt(head, fa, pa, a["context"]["rgb"], zero_a, args.step_size, 1e-3)
            cached.append((fa.detach(), delta_a, fp.detach(), pp, ap["context"]["rgb"], fq.detach(), pq, qi))

        rows = []
        for index, (_, _, current_features, current_prior, current_images, query_features, query_prior, query_images) in enumerate(cached):
            base = torch.zeros(current_features.shape[:3], device="cuda", dtype=current_features.dtype)
            current_atom, _ = adapt(head, current_features, current_prior, current_images, base, args.step_size, 1e-3)
            current = future_loss(head, current_features, current_atom, query_features, query_prior, query_images,
                                  args.temperature, 1e-3)
            candidate_losses = []
            for source_features, source_atom, _, _, _, _, _, _ in cached:
                carried = transport(source_features, source_atom, current_features, args.temperature)
                adapted, _ = adapt(head, current_features, current_prior, current_images, carried,
                                   args.step_size, 1e-3)
                candidate_losses.append(future_loss(head, current_features, adapted, query_features, query_prior,
                                                    query_images, args.temperature, 1e-3))
            values = torch.tensor(candidate_losses)
            positive = candidate_losses[index]
            foreign = [value for j, value in enumerate(candidate_losses) if j != index]
            rank = int((values.argsort() == index).nonzero()[0]) + 1
            rows.append({"positive_rank": rank, "current": current, "matched": positive,
                         "mean_foreign": sum(foreign) / len(foreign),
                         "matched_minus_current": positive - current,
                         "matched_minus_foreign": positive - sum(foreign) / len(foreign)})
            print(json.dumps(rows[-1]))
    summary = {
        "utility_top1": sum(row["positive_rank"] == 1 for row in rows) / len(rows),
        "utility_recall_at_3": sum(row["positive_rank"] <= 3 for row in rows) / len(rows),
        "mean_positive_rank": sum(row["positive_rank"] for row in rows) / len(rows),
        "matched_minus_current": sum(row["matched_minus_current"] for row in rows) / len(rows),
        "matched_minus_foreign": sum(row["matched_minus_foreign"] for row in rows) / len(rows),
    }
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "rows": rows, "step_size": args.step_size,
                   "temperature": args.temperature}, handle, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
