#!/usr/bin/env python3
"""Freeze one metadata-only direction per EXP-009 test overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--config", default="configs/EXP-009_final_test_v24.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source"]["full_manifest"])
    output = Path(config["data"]["manifest"])
    audit_path = Path(config["output"]["pilot_audit"])
    if output.exists() or audit_path.exists():
        raise RuntimeError("EXP-009 test pilot output already exists")
    rows = [row for row in json.loads(source.read_text()) if row.get("split") == "test"]
    if len(rows) != 234:
        raise RuntimeError("frozen EXP-009 test must contain 234 directional episodes")
    by_edge = {}
    for row in rows:
        edge = tuple(sorted((row["source_scene"], row["target_scene"])))
        by_edge.setdefault(edge, []).append(row)
    if len(by_edge) != 117 or any(len(group) != 2 for group in by_edge.values()):
        raise RuntimeError("test directions do not form 117 undirected edges")
    selected = []
    for edge, group in sorted(by_edge.items()):
        selected.append(next(row for row in group if row["source_scene"] == edge[0]))
    components = sorted({int(row["component_id"]) for row in selected})
    locations = sorted({row["location"] for row in selected})
    if len(components) != 22 or len(locations) != 4:
        raise RuntimeError("test pilot lost component/location coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2))
    audit = {
        "experiment": "EXP-009", "stage": "stage15_test_pilot_freeze",
        "protocol_revision": config["protocol_revision"], "split": "test",
        "metadata_only": True, "image_pixels_accessed": False,
        "model_output_accessed": False, "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "pilot_manifest": str(output), "pilot_manifest_sha256": _sha256(output),
        "directional_episodes": len(selected), "components": len(components),
        "component_ids": components, "locations": locations,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
