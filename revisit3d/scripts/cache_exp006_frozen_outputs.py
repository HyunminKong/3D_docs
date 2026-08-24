#!/usr/bin/env python3
"""Cache frozen train-only outputs required by EXP-006 Stage 1/2."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import require_exp006_split
from revisit3d.losses import relative_w2c_from_twist
from revisit3d.models import backproject_tokens, build_geometry_head, local_knn_scale
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid


def _release(module: torch.nn.Module) -> None:
    module.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def _segment_parts(sample: dict) -> list[tuple[str, str, dict]]:
    return [
        ("a_context", sample["a"]["scene"], sample["a"]["context"]),
        ("b_context", sample["b"]["scene"], sample["b"]["context"]),
        ("a_prime_context", sample["a_prime"]["scene"], sample["a_prime"]["context"]),
        ("a_prime_query", sample["a_prime"]["scene"], sample["a_prime"]["query"]),
    ]


def _to_device(part: dict, device: torch.device) -> dict:
    return {key: value.unsqueeze(0).to(device) for key, value in part.items()}


def _geometry_pass(dataset: RevisitEpisodeDataset, config: dict, device: torch.device) -> list[dict]:
    foundation = config["foundation"]
    checkpoint = torch.load(config["stage0"]["output_checkpoint"], map_location="cpu", weights_only=False)
    if not checkpoint.get("health_gate", {}).get("passed") or checkpoint.get("runtime_teacher") is not False:
        raise RuntimeError("Stage-1 cache requires a passed deployable Stage-0 checkpoint")
    extractor = FrozenVGGTFeatures(foundation["checkpoint"], repo_root=foundation["repository"]).to(device)
    head = build_geometry_head(checkpoint["head_type"], extractor.feature_dim).to(device)
    head.load_state_dict(checkpoint["head"])
    head.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            cached = {"episode_id": sample["episode_id"], "segments": {}}
            for tag, scene, raw in _segment_parts(sample):
                part = _to_device(raw, device)
                features = extractor(part["rgb"])
                state = head.initial_state(1, device=device, dtype=features.dtype)
                prediction = head(features, state)
                depth_token = prediction["depth"]
                side = int(math.sqrt(depth_token.shape[2]))
                depth_grid = depth_token.squeeze(-1).reshape(1, depth_token.shape[1], side, side)
                predicted_w2c = relative_w2c_from_twist(prediction["relative_pose"])
                xyz = backproject_tokens(
                    depth_grid, part["intrinsics"], predicted_w2c, image_size=tuple(part["rgb"].shape[-2:])
                )
                scale = local_knn_scale(xyz)
                cached["segments"][tag] = {
                    "scene": scene,
                    "features": features.cpu().half(),
                    "rgb_uint8": (part["rgb"].cpu().clamp(0, 1) * 255).round().to(torch.uint8),
                    "intrinsics": part["intrinsics"].cpu(),
                    "base_depth": depth_grid.cpu(),
                    "base_confidence": prediction["confidence"].cpu().half(),
                    "relative_pose": prediction["relative_pose"].cpu(),
                    "predicted_w2c": predicted_w2c.cpu(),
                    "xyz": xyz.cpu(),
                    "scale": scale.cpu(),
                    "image_size": tuple(part["rgb"].shape[-2:]),
                }
            rows.append(cached)
            print(json.dumps({"cache": "geometry", "index": index, "episode": sample["episode_id"]}), flush=True)
    _release(extractor)
    _release(head)
    return rows


def _tracker_pass(dataset: RevisitEpisodeDataset, rows: list[dict], config: dict, device: torch.device) -> None:
    foundation = config["foundation"]
    side = int(config["stage0"]["track_side"])
    tracker = FrozenVGGTGeometryTracker(foundation["checkpoint"], repo_root=foundation["repository"]).to(device)
    for index, sample in enumerate(dataset):
        for tag, _, raw in _segment_parts(sample):
            part = _to_device(raw, device)
            images = part["rgb"]
            prior = tracker(images, query_grid(images.shape[-2], images.shape[-1], side, str(device)))
            rows[index]["segments"][tag].update({
                "track": prior["track"].cpu().half(),
                "track_visibility": prior["visibility"].cpu().half(),
                "track_confidence": prior["confidence"].cpu().half(),
            })
        print(json.dumps({"cache": "tracker", "index": index, "episode": sample["episode_id"]}), flush=True)
    _release(tracker)


def _fit_pca(rows: list[dict], config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    stage1 = config["stage1"]
    context_tags = ("a_context", "b_context", "a_prime_context")
    matrix = torch.cat([
        row["segments"][tag]["features"].flatten(0, 2)
        for row in rows for tag in context_tags
    ], dim=0)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    sample_size = min(int(stage1["pca_sample_size"]), matrix.shape[0])
    selection = torch.randperm(matrix.shape[0], generator=generator)[:sample_size]
    sample = matrix[selection].to(device=device, dtype=torch.float32)
    sample = F.layer_norm(sample, (sample.shape[-1],))
    mean = sample.mean(dim=0)
    centered = sample - mean
    torch.manual_seed(int(config["seed"]))
    _, _, vectors = torch.pca_lowrank(
        centered, q=64, center=False, niter=int(stage1["pca_iterations"]),
    )
    components = vectors.transpose(0, 1).contiguous()
    if components.shape != (64, int(config["foundation"]["feature_dim"])):
        raise RuntimeError(f"unexpected PCA component shape {tuple(components.shape)}")
    return mean.cpu(), components.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_atom_utility.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 frozen-output cache requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    require_exp006_split(config["data"]["split"])
    cache_path = Path(config["stage1"]["cache"])
    if cache_path.exists() and not args.rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("protocol_revision") != config["protocol_revision"]:
            raise RuntimeError("existing Stage-1 cache uses a different protocol revision; pass --rebuild")
        print(json.dumps({"cache": str(cache_path), "rows": len(payload["rows"]), "reused": True}))
        return
    device = torch.device("cuda")
    data = config["data"]
    dataset = RevisitEpisodeDataset(
        data["manifest"], data["scene_root"], split="train",
        image_size=(int(data["image_height"]), int(data["image_width"])),
    )
    rows = _geometry_pass(dataset, config, device)
    _tracker_pass(dataset, rows, config, device)
    pca_mean, pca_components = _fit_pca(rows, config, device)
    payload = {
        "experiment": "EXP-006",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "stage0_checkpoint": config["stage0"]["output_checkpoint"],
        "pca_mean": pca_mean,
        "pca_components": pca_components,
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(json.dumps({
        "cache": str(cache_path), "rows": len(rows), "pca_samples": int(config["stage1"]["pca_sample_size"]),
        "bytes": cache_path.stat().st_size,
    }))


if __name__ == "__main__":
    main()
