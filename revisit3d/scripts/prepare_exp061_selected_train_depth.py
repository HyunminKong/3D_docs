#!/usr/bin/env python3
"""Register only the preregistered EXP-061 train depths into the RGB frame."""
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
        "--config", default="configs/EXP-061_gauge_local_error_anatomy_v10.yaml"
    )
    parser.add_argument("--confirm-selected-train-depth-registration", action="store_true")
    args = parser.parse_args()
    if not args.confirm_selected_train_depth_registration:
        raise SystemExit("EXP-061 train-depth preparation requires confirmation")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["depth_preparation"])
    if output.exists():
        raise RuntimeError("EXP-061 depth preparation artifact already exists")
    manifest_path = Path(config["data"]["train_manifest"])
    manifest = json.loads(manifest_path.read_text())
    sequences = config["data"]["sequences"]
    if not (
        manifest["role"] == "train"
        and _sha256(manifest_path) == config["data"]["train_manifest_sha256"]
        and set(sequences).issubset(
            {item["relative_path"] for item in manifest["sequences"]}
        )
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-061 train-only depth contract failed")

    fastvggt = Path("FastVGGT").resolve()
    if str(fastvggt) not in sys.path:
        sys.path.insert(0, str(fastvggt))
    from tools.prepare_7scenes import register_depth

    root = Path(config["data"]["root"])
    frames = sorted(
        {
            (sequence, index)
            for sequence in sequences
            for target in config["data"]["target_frames"]
            for index in range(
                int(target) - int(config["data"]["context_frames"]) + 1,
                int(target) + 1,
            )
        }
    )
    rows = []
    for sequence, index in frames:
        raw = root / sequence / f"frame-{index:06d}.depth.png"
        registered = root / sequence / f"frame-{index:06d}.depth.proj.png"
        if not raw.is_file():
            raise RuntimeError(f"EXP-061 raw train depth is missing: {raw}")
        generated = not registered.exists()
        if generated:
            projected = register_depth(io.imread(raw))
            io.imsave(registered, projected, check_contrast=False)
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
        "experiment": "EXP-061",
        "stage": "selected_train_depth_registration",
        "config": str(config_path),
        "selected_frames": len(rows),
        "generated_frames": sum(row["generated"] for row in rows),
        "reused_preexisting_frames": sum(not row["generated"] for row in rows),
        "all_sequences_in_train_role": True,
        "validation_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
