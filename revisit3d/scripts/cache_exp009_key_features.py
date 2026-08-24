#!/usr/bin/env python3
"""Cache frozen VGGT and DINOv2 descriptors for the train-only EXP-009 key pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from revisit3d.backbones import FrozenVGGTFeatures


def _identifier(segment: dict) -> str:
    payload = f"{segment['scene']}:{','.join(map(str, segment['frames']))}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _images(segment: dict, scene_root: Path, size: tuple[int, int]) -> torch.Tensor:
    meta = json.loads((scene_root / segment["scene"] / "opencv_cameras.json").read_text())
    rows = []
    for index in segment["frames"]:
        path = Path(meta["frames"][int(index)]["file_path"])
        image = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
        rows.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_key_backbone_v14.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["feature_cache"])
    if output.exists():
        raise RuntimeError(f"EXP-009 key cache already exists: {output}")
    pairs = json.loads(Path(config["data"]["pair_manifest"]).read_text())
    if not pairs or any(row.get("location") is None for row in pairs):
        raise RuntimeError("invalid train-only key pilot")
    segments = {}
    for row in pairs:
        for side in ("left", "right"):
            segment = row[side]
            segments.setdefault(_identifier(segment), segment)
    scene_root = Path(config["data"]["scene_root"])
    size = (int(config["data"]["image_height"]), int(config["data"]["image_width"]))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 key feature extraction requires CUDA")
    rows = {key: {"segment": value} for key, value in segments.items()}

    vggt_config = config["models"]["vggt"]
    vggt = FrozenVGGTFeatures(
        vggt_config["checkpoint"], repo_root=vggt_config["repository"],
    ).to(device).eval()
    for position, (key, segment) in enumerate(sorted(segments.items())):
        images = _images(segment, scene_root, size).unsqueeze(0).to(device)
        with torch.inference_mode():
            tokens = vggt(images)
            descriptor = F.normalize(tokens.mean(dim=2)[0], dim=-1)
        rows[key]["vggt"] = descriptor.cpu().half()
        if (position + 1) % 25 == 0 or position + 1 == len(segments):
            print(json.dumps({"model": "vggt", "completed": position + 1, "total": len(segments)}), flush=True)
    del vggt
    gc.collect()
    torch.cuda.empty_cache()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import timm

    dino_name = config["models"]["dinov2"]["timm_name"]
    dino = timm.create_model(dino_name, pretrained=True, num_classes=0, img_size=size[0])
    dino.eval().requires_grad_(False).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    for position, (key, segment) in enumerate(sorted(segments.items())):
        images = _images(segment, scene_root, size).to(device)
        images = (images - mean) / std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = dino.forward_features(images)
            if not isinstance(features, dict) or "x_norm_clstoken" not in features:
                raise RuntimeError("unexpected DINOv2 forward_features contract")
            descriptor = F.normalize(features["x_norm_clstoken"].float(), dim=-1)
        rows[key]["dinov2"] = descriptor.cpu().half()
        if (position + 1) % 25 == 0 or position + 1 == len(segments):
            print(json.dumps({"model": "dinov2", "completed": position + 1, "total": len(segments)}), flush=True)
    del dino
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "experiment": "EXP-009", "protocol_revision": config["protocol_revision"],
        "split": "train", "validation_accessed": False, "test_accessed": False,
        "config": str(config_path), "pair_manifest": config["data"]["pair_manifest"],
        "models": config["models"], "segments": len(rows), "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "segments": len(rows), "bytes": output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
