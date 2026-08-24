#!/usr/bin/env python3
"""Freeze one directional geometry episode for each EXP-009 key-pilot positive edge."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
    parser.add_argument("--config", default="configs/EXP-009_locked_transfer_v15.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["data"]["manifest"])
    result_path = Path(config["data"]["selection_result"])
    if output.exists() or result_path.exists():
        raise RuntimeError("EXP-009 geometry pilot already exists")
    source = json.loads(Path(config["data"]["source_manifest"]).read_text())
    key_pairs = json.loads(Path(config["data"]["key_pair_manifest"]).read_text())
    source_by_id = {row["episode_id"]: row for row in source}
    selected = []
    for pair in key_pairs:
        if int(pair["label"]) != 1:
            continue
        episode_id = f"{pair['left']['scene']}__{pair['right']['scene']}"
        row = source_by_id[episode_id]
        if row["split"] != "train" or int(row["component_id"]) != int(pair["component_id"]):
            raise RuntimeError("key positive disagrees with frozen train episode")
        selected.append(row)
    selected.sort(key=lambda row: row["episode_id"])
    if len(selected) != 225 or len({row["episode_id"] for row in selected}) != len(selected):
        raise RuntimeError("geometry pilot must contain 225 unique train episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage5_geometry_pilot_selection",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "metadata_only": True, "image_pixels_accessed": False, "model_output_accessed": False,
        "manifest": str(output), "manifest_sha256": _sha256(output),
        "episodes": len(selected),
        "components": len({row["component_id"] for row in selected}),
        "locations": dict(Counter(row["location"] for row in selected)),
        "validation_accessed": False, "test_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
