#!/usr/bin/env python3
"""Frozen RGB-only FastVGGT cache for the EXP-035 TUM transfer test."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.losses import relative_w2c_from_twist
from revisit3d.models import backproject_tokens, build_geometry_head, local_knn_scale
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _load_views(frames: list[dict], *, height: int, width: int) -> dict:
    rgbs, intrinsics = [], []
    for frame in frames:
        image = Image.open(frame["rgb"]).convert("RGB")
        old_width, old_height = image.size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        fx, fy, cx, cy = frame["intrinsics_fx_fy_cx_cy"]
        intrinsics.append(
            torch.tensor(
                [
                    fx * width / old_width,
                    fy * height / old_height,
                    cx * width / old_width,
                    cy * height / old_height,
                ],
                dtype=torch.float32,
            )
        )
        rgbs.append(rgb)
    return {"rgb": torch.stack(rgbs).unsqueeze(0), "intrinsics": torch.stack(intrinsics).unsqueeze(0)}


def _frames(event: dict, tag: str) -> list[dict]:
    intrinsics = event["intrinsics_fx_fy_cx_cy"]
    return [{**frame, "intrinsics_fx_fy_cx_cy": intrinsics} for frame in event[tag]]


def _release(module: torch.nn.Module) -> None:
    module.cpu()
    del module
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-035_tum_zero_shot_transfer_v10.yaml")
    parser.add_argument("--confirm-zero-shot-access", action="store_true")
    args = parser.parse_args()
    if not args.confirm_zero_shot_access:
        raise SystemExit("EXP-035 RGB/model access requires explicit confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-035 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    cache_path = Path(config["stage1"]["cache"])
    result_path = Path(config["stage1"]["result"])
    if cache_path.exists() or result_path.exists():
        raise RuntimeError("EXP-035 Stage-1 output already exists")
    authorization = json.loads(Path(config["authorization"]).read_text())
    manifest_path = Path(config["data"]["manifest"])
    if not (
        authorization["registered_gate"]["passed"]
        and not authorization["image_decoded"]
        and not authorization["depth_decoded"]
        and not authorization["model_output_accessed"]
        and _sha256(manifest_path) == config["data"]["manifest_sha256"]
    ):
        raise RuntimeError("EXP-035 authorization contract failed")
    events = json.loads(manifest_path.read_text())
    if len(events) != 223:
        raise RuntimeError("TUM stream manifest count changed")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    height = int(config["data"]["image_height"])
    width = int(config["data"]["image_width"])
    geometry_checkpoint = torch.load(
        config["geometry_head_checkpoint"], map_location="cpu", weights_only=False
    )
    extractor = FrozenVGGTFeatures(
        config["foundation"]["checkpoint"],
        repo_root=config["foundation"]["repository"],
    ).to(device)
    geometry_head = build_geometry_head(
        geometry_checkpoint["head_type"], extractor.feature_dim
    ).to(device)
    geometry_head.load_state_dict(geometry_checkpoint["head"])
    geometry_head.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for index, event in enumerate(events):
            row = {"event_id": event["event_id"], "sequence": event["sequence"], "segments": {}}
            for tag in ("context", "query"):
                raw = _load_views(_frames(event, tag), height=height, width=width)
                rgb = raw["rgb"].to(device)
                intrinsics = raw["intrinsics"].to(device)
                features = extractor(rgb)
                state = geometry_head.initial_state(1, device=device, dtype=features.dtype)
                prediction = geometry_head(features, state)
                side = int(math.sqrt(prediction["depth"].shape[2]))
                depth = prediction["depth"].squeeze(-1).reshape(1, rgb.shape[1], side, side)
                w2c = relative_w2c_from_twist(prediction["relative_pose"])
                xyz = backproject_tokens(depth, intrinsics, w2c, image_size=(height, width))
                scale = local_knn_scale(xyz)
                row["segments"][tag] = {
                    "scene": event["sequence"],
                    "features": features.cpu().half(),
                    "rgb_uint8": (rgb.cpu().clamp(0, 1) * 255).round().to(torch.uint8),
                    "intrinsics": intrinsics.cpu(),
                    "base_depth": depth.cpu(),
                    "base_confidence": prediction["confidence"].cpu().half(),
                    "relative_pose": prediction["relative_pose"].cpu(),
                    "predicted_w2c": w2c.cpu(),
                    "xyz": xyz.cpu(),
                    "scale": scale.cpu(),
                    "image_size": (height, width),
                }
            rows.append(row)
            if index % 10 == 0 or index + 1 == len(events):
                print(json.dumps({"stage": "geometry", "row": index + 1, "total": len(events)}), flush=True)
    _release(extractor)
    _release(geometry_head)

    tracker = FrozenVGGTGeometryTracker(
        config["foundation"]["checkpoint"],
        repo_root=config["foundation"]["repository"],
    ).to(device)
    side = int(config["foundation"]["track_side"])
    with torch.no_grad():
        for index, event in enumerate(events):
            for tag in ("context", "query"):
                raw = _load_views(_frames(event, tag), height=height, width=width)
                rgb = raw["rgb"].to(device)
                prior = tracker(rgb, query_grid(height, width, side, str(device)))
                rows[index]["segments"][tag].update(
                    {
                        "track": prior["track"].cpu().half(),
                        "track_visibility": prior["visibility"].cpu().half(),
                        "track_confidence": prior["confidence"].cpu().half(),
                    }
                )
            if index % 10 == 0 or index + 1 == len(events):
                print(json.dumps({"stage": "tracker", "row": index + 1, "total": len(events)}), flush=True)
    _release(tracker)

    payload = {
        "experiment": "EXP-035",
        "protocol_revision": config["protocol_revision"],
        "split": "tum_zero_shot",
        "manifest_sha256": config["data"]["manifest_sha256"],
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    summary = {
        "experiment": "EXP-035",
        "stage": "tum_frozen_geometry_cache",
        "protocol_revision": config["protocol_revision"],
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "rows": len(rows),
        "image_decoded": True,
        "model_output_accessed": True,
        "depth_decoded": False,
        "tum_fit_performed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
