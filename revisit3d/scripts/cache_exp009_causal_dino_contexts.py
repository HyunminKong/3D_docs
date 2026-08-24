#!/usr/bin/env python3
"""Cache DINOv2 view-token sets for unique train contexts in EXP-009 Stage 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F


def _identifier(segment: dict) -> str:
    payload = f"{segment['scene']}:{','.join(map(str, segment['frames']))}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _four_view(segment: dict) -> dict:
    frames = [int(segment["frames"][index]) for index in (0, 2, 5, 7)]
    return {"scene": segment["scene"], "frames": frames}


def _images(segment: dict, scene_root: Path, size: tuple[int, int]) -> torch.Tensor:
    metadata = json.loads((scene_root / segment["scene"] / "opencv_cameras.json").read_text())
    rows = []
    for index in segment["frames"]:
        path = Path(metadata["frames"][int(index)]["file_path"])
        image = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
        rows.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_causal_dino_retrieval_v17.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["dinov2_context_cache"])
    if output.exists():
        raise RuntimeError(f"EXP-009 causal DINO cache already exists: {output}")
    if config["data"]["split"] != "train":
        raise RuntimeError("Stage 7 feature extraction is train-only")
    manifest = json.loads(Path(config["data"]["geometry_manifest"]).read_text())
    if len(manifest) != 225 or any(row.get("split") != "train" for row in manifest):
        raise RuntimeError("causal DINO cache requires the frozen 225-episode train pilot")

    contexts = {}
    for row in manifest:
        for tag in ("a", "b", "a_prime"):
            segment = row[tag]
            contexts.setdefault(_identifier(segment), segment)
    if len(contexts) != 557:
        raise RuntimeError(f"expected 557 unique full contexts, found {len(contexts)}")

    pair_cache = torch.load(config["data"]["dinov2_pair_cache"], map_location="cpu", weights_only=False)
    if not (
        pair_cache.get("model") == "dinov2"
        and pair_cache.get("split") == "train"
        and pair_cache.get("validation_accessed") is False
        and pair_cache.get("test_accessed") is False
    ):
        raise RuntimeError("Stage-4 DINO cache violates the train-only contract")
    reusable = {}
    for row in pair_cache["rows"].values():
        reusable[_identifier(row["segment"])] = row["dinov2"].float()

    rows = {}
    missing = {}
    for key, segment in sorted(contexts.items()):
        four = _four_view(segment)
        four_key = _identifier(four)
        if four_key in reusable:
            rows[key] = {"segment": segment, "sampled_segment": four, "dinov2": reusable[four_key].half()}
        else:
            missing[key] = (segment, four)

    if len(rows) != 426 or len(missing) != 131:
        raise RuntimeError(
            f"unexpected Stage-4 reuse split: reused={len(rows)}, missing={len(missing)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 causal DINO extraction requires CUDA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import timm

    device = torch.device("cuda")
    size = (int(config["data"]["image_height"]), int(config["data"]["image_width"]))
    model_config = config["models"]["dinov2"]
    model = timm.create_model(
        model_config["timm_name"], pretrained=True, num_classes=0, img_size=size[0],
    )
    model.eval().requires_grad_(False).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    scene_root = Path(config["data"]["scene_root"])
    for position, (key, (segment, four)) in enumerate(sorted(missing.items())):
        images = (_images(four, scene_root, size).to(device) - mean) / std
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.forward_features(images)
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                cls_token = features["x_norm_clstoken"]
            elif isinstance(features, torch.Tensor) and features.ndim == 3:
                cls_token = features[:, 0]
            else:
                raise RuntimeError("unexpected DINOv2 forward_features contract")
            descriptor = F.normalize(cls_token.float(), dim=-1)
        rows[key] = {"segment": segment, "sampled_segment": four, "dinov2": descriptor.cpu().half()}
        if (position + 1) % 25 == 0 or position + 1 == len(missing):
            print(json.dumps({"extracted": position + 1, "total_missing": len(missing)}), flush=True)

    payload = {
        "experiment": "EXP-009", "stage": "stage7_causal_dino_context_cache",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "validation_accessed": False, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "contexts": len(rows), "reused_stage4": 426, "extracted": 131,
        "model_config": model_config, "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"output": str(output), "contexts": len(rows), "bytes": output.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
