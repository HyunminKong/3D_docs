#!/usr/bin/env python3
"""Create the compact, Git-tracked EXP-007 audit summary from local raw JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DROP_KEYS = {
    "rows", "pair_rows", "streams", "contexts", "pairs", "events", "candidates",
    "orders", "folds", "router_features",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact(value):
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in DROP_KEYS}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def main() -> None:
    root = Path("revisit3d/results/EXP-007")
    output = root / "summary_v21.json"
    artifacts = {}
    for path in sorted(root.glob("*.json")):
        if path == output:
            continue
        payload = json.loads(path.read_text())
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "compact_result": _compact(payload),
        }
    result = {
        "experiment": "EXP-007",
        "summary_revision": "v2.1",
        "split": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "raw_artifacts_tracked": False,
        "artifacts": artifacts,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "artifacts": len(artifacts), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
