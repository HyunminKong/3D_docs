#!/usr/bin/env python3
"""Convert only camera metadata for the frozen EXP-009 scene set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_convert_metadata_v12.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    manifest_path = Path(config["data"]["manifest"])
    audit_path = Path(config["output"]["audit"])
    scene_list_path = Path(config["output"]["scene_list"])
    if audit_path.exists():
        raise RuntimeError(f"EXP-009 conversion audit already exists: {audit_path}")
    episodes = json.loads(manifest_path.read_text())
    scenes = sorted({
        scene for row in episodes for scene in (row["source_scene"], row["target_scene"])
    })
    split_scenes = {
        split: sorted({
            scene for row in episodes if row["split"] == split
            for scene in (row["source_scene"], row["target_scene"])
        }) for split in ("train", "val", "test")
    }
    if any(set(split_scenes[left]) & set(split_scenes[right])
           for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("frozen manifest has scene leakage")
    scene_list_path.parent.mkdir(parents=True, exist_ok=True)
    scene_list_path.write_text("\n".join(scenes) + "\n")
    output_root = Path(config["data"]["output_root"])
    command = [
        sys.executable, "tttLRM/oracle/nuscenes_convert.py",
        "--root", config["data"]["nuscenes_root"],
        "--version", config["data"]["nuscenes_version"],
        "--scene-list", str(scene_list_path),
        "--out", str(output_root),
        "--min-frames", str(config["data"]["minimum_frames"]),
    ]
    subprocess.run(command, check=True)
    converted = sorted(path.parent.name for path in output_root.glob("*/opencv_cameras.json"))
    missing = sorted(set(scenes) - set(converted))
    extra = sorted(set(converted) - set(scenes))
    if missing or extra:
        raise RuntimeError(f"converted scene mismatch: missing={len(missing)}, extra={len(extra)}")
    audit = {
        "experiment": "EXP-009", "stage": "stage2_metadata_conversion",
        "protocol_revision": config["protocol_revision"],
        "image_pixels_accessed": False, "model_output_accessed": False,
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "output_root": str(output_root), "converted_scenes": len(converted),
        "split_scenes": {key: len(value) for key, value in split_scenes.items()},
        "missing_scenes": missing, "extra_scenes": extra, "passed": not missing and not extra,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps(audit, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
