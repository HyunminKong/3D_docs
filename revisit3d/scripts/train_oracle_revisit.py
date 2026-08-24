#!/usr/bin/env python3
"""First memory-free A→B→A' meta-training entrypoint.

This is deliberately an oracle-pair training phase.  The manifest supplies the
matched earlier traversal; no search key, bank, or test-time write is present.
It answers the required causal question: can a compact TTT coordinate be made
reusable at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import depth_smoothness_loss, relative_w2c_from_twist, reprojection_loss
from revisit3d.models import RevisitMetaLearner, SignedResidualTransport, StreamingGeometryHead


def depth_grid(prediction: dict[str, torch.Tensor]) -> torch.Tensor:
    depth = prediction["depth"].squeeze(-1)
    patches = depth.shape[-1]
    side = int(patches ** 0.5)
    if side * side != patches:
        raise ValueError(f"token count {patches} is not a square patch grid")
    return depth.reshape(*depth.shape[:2], side, side)


def segment_loss(prediction: dict[str, torch.Tensor], segment: dict, *, smoothness: float,
                 pose_source: str) -> torch.Tensor:
    depth = depth_grid(prediction)
    images = segment["context"]["rgb"]
    intrinsics = segment["context"]["intrinsics"]
    w2c = segment["context"]["w2c"] if pose_source == "known" else relative_w2c_from_twist(prediction["relative_pose"])
    photo = reprojection_loss(depth, images, intrinsics, w2c)
    return photo + smoothness * depth_smoothness_loss(depth, images)


def to_device(segment: dict, device: str) -> dict:
    return {
        "scene": segment["scene"],
        "context": {key: value.unsqueeze(0).to(device) for key, value in segment["context"].items()},
        "query": {key: value.unsqueeze(0).to(device) for key, value in segment["query"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--steps", type=int, default=1, help="outer optimiser steps; use 1 for a plumbing check")
    parser.add_argument("--ttt-steps", type=int, default=1)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--outer-lr", type=float, default=1e-4)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.0,
                        help="weight for matched-vs-foreign anti-collapse loss")
    parser.add_argument("--contrastive-margin", type=float, default=1e-2,
                        help="foreign held-out loss must exceed matched loss by this amount")
    parser.add_argument("--out", default="revisit3d/checkpoints/oracle_bootstrap.pt")
    parser.add_argument("--pose-source", choices=("known", "predicted"), default="known")
    parser.add_argument("--known-pose-bootstrap", action="store_true")
    args = parser.parse_args()
    if args.pose_source == "known" and not args.known_pose_bootstrap:
        raise SystemExit("This bootstrap uses supplied transforms for reprojection. Pass --known-pose-bootstrap explicitly.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split="train", image_size=(224, 224))
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").to(device)
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).to(device)
    transport = SignedResidualTransport(extractor.feature_dim, state_dim=32, hidden_dim=128).to(device)
    learner = RevisitMetaLearner(head, transport)
    optimiser = torch.optim.AdamW(learner.parameters(), lr=args.outer_lr, weight_decay=1e-4)

    records = []
    for outer_step, sample in zip(range(args.steps), loader):
        a, b, ap = (to_device(sample[name], device) for name in ("a", "b", "a_prime"))
        features_a = extractor(a["context"]["rgb"])
        features_b = extractor(b["context"]["rgb"])
        features_ap = extractor(ap["context"]["rgb"])
        features_ap_query = extractor(ap["query"]["rgb"])
        segments = {"a": a, "b": b, "a_prime": ap}
        rollout = learner.rollout(
            features_a, features_b, features_ap,
            lambda prediction, tag: segment_loss(prediction, segments[tag], smoothness=args.smoothness,
                                                  pose_source=args.pose_source),
            features_a_prime_query=features_ap_query,
            steps=args.ttt_steps, learning_rate=args.ttt_lr,
            create_graph=False, retain_state_gradient=True,
        )
        # Queries are deliberately not passed to the TTT callback above.
        loss = learner.revisit_outer_loss(
            rollout, lambda prediction: reprojection_loss(
                depth_grid(prediction), ap["query"]["rgb"], ap["query"]["intrinsics"],
                ap["query"]["w2c"] if args.pose_source == "known"
                else relative_w2c_from_twist(prediction["relative_pose"])
            )
        )
        foreign_loss = None
        if args.contrastive_weight:
            # A train-split episode with no source/target scene in common is a
            # foreign skill.  It is never a retrieval candidate at deployment;
            # it exists solely to prevent the residual map from becoming a
            # global bias that helps every query equally.
            current_scenes = {sample["a"]["scene"], sample["a_prime"]["scene"]}
            foreign_index = next(i for i, record in enumerate(dataset.records)
                                 if not current_scenes.intersection({record["source_scene"], record["target_scene"]}))
            foreign_segment = to_device(dataset[foreign_index]["a"], device)
            foreign_features = extractor(foreign_segment["context"]["rgb"])
            foreign_state = head.adapt(
                foreign_features, head.initial_state(1, device=device, dtype=foreign_features.dtype),
                lambda prediction: segment_loss(prediction, foreign_segment, smoothness=args.smoothness,
                                                 pose_source=args.pose_source),
                steps=args.ttt_steps, learning_rate=args.ttt_lr, retain_state_gradient=True,
            )[0]
            foreign_reuse = head.adapt(
                features_ap,
                type(foreign_state)(rollout.state_ab.value + transport(foreign_state, features_ap).value),
                lambda prediction: segment_loss(prediction, ap, smoothness=args.smoothness,
                                                 pose_source=args.pose_source),
                steps=args.ttt_steps, learning_rate=args.ttt_lr, retain_state_gradient=True,
            )[0]
            foreign_loss = reprojection_loss(
                depth_grid(head(features_ap_query, foreign_reuse)), ap["query"]["rgb"], ap["query"]["intrinsics"],
                ap["query"]["w2c"] if args.pose_source == "known"
                else relative_w2c_from_twist(head(features_ap_query, foreign_reuse)["relative_pose"])
            )
            matched_loss = reprojection_loss(
                depth_grid(rollout.prediction_reuse), ap["query"]["rgb"], ap["query"]["intrinsics"],
                ap["query"]["w2c"] if args.pose_source == "known"
                else relative_w2c_from_twist(rollout.prediction_reuse["relative_pose"])
            )
            loss = loss + args.contrastive_weight * torch.relu(
                args.contrastive_margin - (foreign_loss - matched_loss)
            )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
        optimiser.step()
        row = {"step": outer_step, "episode": sample["episode_id"], "outer_loss": float(loss.detach()),
               "foreign_loss": None if foreign_loss is None else float(foreign_loss.detach())}
        records.append(row)
        print(json.dumps(row))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "transport": transport.state_dict(), "records": records,
                "known_pose_bootstrap": args.pose_source == "known", "pose_source": args.pose_source}, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
