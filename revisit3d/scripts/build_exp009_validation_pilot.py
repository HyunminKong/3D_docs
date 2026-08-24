#!/usr/bin/env python3
"""Freeze one direction per unseen EXP-009 validation overlap using metadata only."""

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
    parser.add_argument("--config", default="configs/EXP-009_validation_lock_v22.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    manifest_path = Path(config["source"]["full_manifest"])
    output = Path(config["data"]["manifest"])
    audit_path = Path(config["output"]["pilot_audit"])
    if output.exists() or audit_path.exists():
        raise RuntimeError("EXP-009 validation pilot output already exists")
    rows = [
        row for row in json.loads(manifest_path.read_text()) if row.get("split") == "val"
    ]
    if len(rows) != 234:
        raise RuntimeError("frozen EXP-009 validation must contain 234 directional episodes")
    by_edge = {}
    for row in rows:
        edge = tuple(sorted((row["source_scene"], row["target_scene"])))
        by_edge.setdefault(edge, []).append(row)
    if len(by_edge) != 117 or any(len(group) != 2 for group in by_edge.values()):
        raise RuntimeError("validation directional pairs do not form 117 undirected edges")
    selected = []
    for edge, group in sorted(by_edge.items()):
        choice = next(
            (row for row in group if row["source_scene"] == edge[0]), None,
        )
        if choice is None:
            raise RuntimeError(f"missing canonical direction for edge {edge}")
        selected.append(choice)
    components = sorted({int(row["component_id"]) for row in selected})
    locations = sorted({row["location"] for row in selected})
    if len(components) != 17 or len(locations) != 4:
        raise RuntimeError("validation pilot lost component/location coverage")
    if any(row.get("split") != "val" for row in selected):
        raise RuntimeError("validation pilot contains a non-validation row")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2))
    audit = {
        "experiment": "EXP-009", "stage": "stage12_validation_pilot_freeze",
        "protocol_revision": config["protocol_revision"], "split": "val",
        "metadata_only": True, "image_pixels_accessed": False,
        "model_output_accessed": False, "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "pilot_manifest": str(output), "pilot_manifest_sha256": _sha256(output),
        "directional_episodes": len(selected), "components": len(components),
        "component_ids": components, "locations": locations,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
