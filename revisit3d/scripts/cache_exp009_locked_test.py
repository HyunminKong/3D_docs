#!/usr/bin/env python3
"""Explicit one-shot frozen-output cache for the locked EXP-009 test pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from revisit3d.data import RevisitEpisodeDataset
from revisit3d.scripts.cache_exp006_frozen_outputs import _geometry_pass, _tracker_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_final_test_v24.yaml")
    parser.add_argument("--confirm-locked-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_locked_test:
        raise SystemExit("refusing test pixel access without explicit final-lock confirmation")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-009 locked test cache requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["stage1"]["cache"])
    if output.exists():
        raise RuntimeError("locked test cache already exists")
    lock = json.loads(Path(config["output"]["lock_result"]).read_text())
    audit = json.loads(Path(config["output"]["pilot_audit"]).read_text())
    if not (
        lock.get("test_accessed") is False
        and lock.get("selected_capacity") == 64
        and audit.get("split") == "test"
        and audit.get("image_pixels_accessed") is False
        and audit.get("directional_episodes") == 117
        and audit.get("components") == 22
    ):
        raise RuntimeError("test cache requested before complete final lock")
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"], config["data"]["scene_root"], split="test",
        image_size=(int(config["data"]["image_height"]), int(config["data"]["image_width"])),
    )
    if len(dataset) != 117:
        raise RuntimeError("final test pilot must contain 117 episodes")
    device = torch.device("cuda")
    rows = _geometry_pass(dataset, config, device)
    _tracker_pass(dataset, rows, config, device)
    source_path = Path(config["stage1"]["pca_source_cache"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("split") != "train" or source.get("protocol_revision") != "v1.5":
        raise RuntimeError("final test PCA must come from the locked EXP-009 train cache")
    payload = {
        "experiment": "EXP-009", "protocol_revision": config["protocol_revision"],
        "split": "test", "test_access_authorized_by": config["output"]["lock_result"],
        "pca_fit_split": "train", "pca_source_cache": str(source_path),
        "pca_mean": source["pca_mean"], "pca_components": source["pca_components"],
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "cache": str(output), "rows": len(rows), "split": "test",
        "pca_fit_split": "train", "bytes": output.stat().st_size,
    }), flush=True)


if __name__ == "__main__":
    main()
