#!/usr/bin/env python3
"""Diagnose whether the proposed online objective supplies a usable TTT signal.

This is a *pre-framework* diagnostic.  It intentionally reports both a
known-pose counterfactual and the deployment-style predicted-pose objective:
if the latter has vanishing state gradients or predicts near-identity motion,
then a memory result cannot be interpreted as evidence about revisit reuse.
"""

from __future__ import annotations

import argparse
import json

import torch

from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.losses import relative_w2c_from_twist
from revisit3d.models import StreamingGeometryHead
from revisit3d.scripts.train_oracle_revisit import depth_grid, segment_loss, to_device


def state_gradient(head, features, segment, pose_source: str, smoothness: float) -> tuple[float, float, float, float]:
    state = head.initial_state(1, device=features.device, dtype=features.dtype)
    state.value.requires_grad_(True)
    prediction = head(features, state)
    loss = segment_loss(prediction, segment, smoothness=smoothness, pose_source=pose_source)
    gradient, = torch.autograd.grad(loss, state.value)
    pose = prediction["relative_pose"][:, 1:]
    translation = pose[..., :3].norm(dim=-1).mean()
    rotation = pose[..., 3:].norm(dim=-1).mean()
    return float(loss.detach()), float(gradient.norm().detach()), float(translation.detach()), float(rotation.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="revisit3d/manifests/nuscenes_revisit.json")
    parser.add_argument("--scene-root", default="tttLRM/data_example/nuscenes_2x2")
    parser.add_argument("--vggt-checkpoint", default="FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--ttt-lr", type=float, default=1e-2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RevisitEpisodeDataset(args.manifest, args.scene_root, split=args.split, image_size=(224, 224))
    extractor = FrozenVGGTFeatures(args.vggt_checkpoint, repo_root="FastVGGT").to(device)
    head = StreamingGeometryHead(extractor.feature_dim, state_dim=32, hidden_dim=512).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head"])
    head.eval()

    rows = []
    for sample in dataset:
        a = to_device(sample["a"], device)
        features = extractor(a["context"]["rgb"])
        with torch.enable_grad():
            known = state_gradient(head, features, a, "known", args.smoothness)
            predicted = state_gradient(head, features, a, "predicted", args.smoothness)
            initial = head.initial_state(1, device=device, dtype=features.dtype)
            adapted, _ = head.adapt(
                features, initial,
                lambda prediction: segment_loss(prediction, a, smoothness=args.smoothness, pose_source="predicted"),
                steps=1, learning_rate=args.ttt_lr,
            )
            state_step = (adapted.value - initial.value).norm()
        with torch.no_grad():
            prediction = head(features, initial)
            depth = depth_grid(prediction)
            pose_w2c = relative_w2c_from_twist(prediction["relative_pose"])
            relative = pose_w2c[:, 1:, :3, 3].norm(dim=-1).mean() if pose_w2c.shape[1] > 1 else depth.new_zeros(())
        rows.append({
            "episode": sample["episode_id"],
            "known_loss": known[0], "known_grad_norm": known[1],
            "predicted_loss": predicted[0], "predicted_grad_norm": predicted[1],
            "raw_pose_translation_norm": predicted[2], "raw_pose_rotation_norm": predicted[3],
            "relative_translation_norm": float(relative),
            "depth_mean": float(depth.mean()), "depth_std": float(depth.std()),
            "one_step_state_delta_norm": float(state_step),
        })
        print(json.dumps(rows[-1]))

    keys = [key for key in rows[0] if key != "episode"]
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    payload = {"checkpoint": args.checkpoint, "split": args.split, "rows": rows, "summary": summary}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}))


if __name__ == "__main__":
    main()
