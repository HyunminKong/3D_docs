#!/usr/bin/env python3
"""Freeze the current TUM RGB-D bytes used by the EXP-050 correction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="revisit3d/manifests/tum_zero_shot_stream_exp034_v11.json"
    )
    parser.add_argument(
        "--output", default="revisit3d/results/EXP-050/tum_input_inventory_v11.json"
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("EXP-050 input inventory already exists")
    events = json.loads(manifest.read_text())
    paths = sorted(
        {
            str(Path(frame[key]).resolve())
            for event in events
            for frame in (*event["context"], *event["query"])
            for key in ("rgb", "depth")
        }
    )
    digest = hashlib.sha256()
    total_bytes = 0
    by_suffix: dict[str, int] = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"EXP-050 input is missing: {path}")
        file_hash = _sha256(path)
        size = path.stat().st_size
        total_bytes += size
        by_suffix[path.suffix] = by_suffix.get(path.suffix, 0) + 1
        digest.update(value.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    result = {
        "experiment": "EXP-050",
        "stage": "tum_rgbd_input_inventory_v11",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "unique_files": len(paths),
        "files_by_suffix": by_suffix,
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
        "file_contents_decoded": False,
        "model_accessed": False,
        "terminal_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
