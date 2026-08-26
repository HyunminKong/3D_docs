#!/usr/bin/env python3
"""Register only EXP-067's preregistered train-pair depths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from skimage import io

from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-067_function_space_plasticity_transport_v10.yaml"
    )
    parser.add_argument("--confirm-selected-train-depth-registration", action="store_true")
    args = parser.parse_args()
    if not args.confirm_selected_train_depth_registration:
        raise SystemExit("EXP-067 train-depth preparation requires confirmation")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["depth_preparation"])
    if output.exists():
        raise RuntimeError("EXP-067 depth preparation artifact already exists")
    manifest_path = Path(config["data"]["train_manifest"])
    manifest = json.loads(manifest_path.read_text())
    sequences = {pair["sequence"] for pair in config["data"]["pairs"]}
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and sequences.issubset({item["relative_path"] for item in manifest["sequences"]})
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-067 train-only depth contract failed")
    fastvggt = Path("FastVGGT").resolve()
    if str(fastvggt) not in sys.path:
        sys.path.insert(0, str(fastvggt))
    from tools.prepare_7scenes import register_depth

    frames = sorted(
        {
            (pair["sequence"], int(frame) + offset)
            for pair in config["data"]["pairs"]
            for frame in (pair["source"], pair["target"])
            for offset in (-1, 0)
        }
    )
    root = Path(config["data"]["root"])
    rows = []
    for sequence, index in frames:
        raw = root / sequence / f"frame-{index:06d}.depth.png"
        registered = root / sequence / f"frame-{index:06d}.depth.proj.png"
        if not raw.is_file():
            raise RuntimeError(f"EXP-067 raw train depth is missing: {raw}")
        generated = not registered.exists()
        if generated:
            io.imsave(registered, register_depth(io.imread(raw)), check_contrast=False)
        rows.append(
            {
                "sequence": sequence,
                "frame": index,
                "generated": generated,
                "raw_sha256": _sha256(raw),
                "registered_sha256": _sha256(registered),
            }
        )
    result = {
        "experiment": "EXP-067",
        "stage": "selected_train_depth_registration",
        "config": str(config_path),
        "selected_frames": len(rows),
        "generated_frames": sum(row["generated"] for row in rows),
        "reused_preexisting_frames": sum(not row["generated"] for row in rows),
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
