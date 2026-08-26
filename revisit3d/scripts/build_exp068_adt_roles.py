#!/usr/bin/env python3
"""Freeze EXP-068 ADT roles from names only, before NPZ/model access."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(name: str) -> str:
    if "decoration_skeleton" in name:
        return "decoration_skeleton"
    if "decoration" in name:
        return "decoration"
    if "golden_skeleton" in name:
        return "golden_skeleton"
    if "meal" in name:
        return "meal"
    if "clean" in name:
        return "clean"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-068_cross_clip_relational_consistency_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    root = Path(config["data"]["root"])
    output = Path(config["data"]["manifest"])
    if output.exists():
        raise RuntimeError(f"EXP-068 role manifest already exists: {output}")

    exposed = set(config["data"]["exposed_external_sequences"])
    files = sorted(root.glob("*.npz"))
    names = [path.stem for path in files]
    if len(files) != 49 or not exposed.issubset(names):
        raise RuntimeError("EXP-068 expected exactly 49 ADT files and all ten exposed names")
    fresh = [name for name in names if name not in exposed]
    if len(fresh) != 39:
        raise RuntimeError(f"EXP-068 expected 39 fresh names, found {len(fresh)}")

    # Sort within semantic filename families by a stable name hash, then deal
    # round-robin into roles. This avoids an alphabetical family/domain split
    # while never reading NPZ metadata, pixels, tracks, or model outputs.
    by_family: dict[str, list[str]] = defaultdict(list)
    for name in fresh:
        by_family[family(name)].append(name)
    ordered: list[str] = []
    for fam in sorted(by_family):
        ordered.extend(
            sorted(
                by_family[fam],
                key=lambda value: hashlib.sha256(f"EXP-068::{value}".encode()).hexdigest(),
            )
        )

    target_counts = {key: int(value) for key, value in config["data"]["roles"].items()}
    roles = {key: [] for key in ("premise", "validation", "terminal")}
    cycle = ("premise", "validation", "terminal")
    cursor = 0
    for name in ordered:
        for _ in range(len(cycle)):
            role = cycle[cursor % len(cycle)]
            cursor += 1
            if len(roles[role]) < target_counts[role]:
                roles[role].append(name)
                break
        else:
            raise RuntimeError("EXP-068 role allocation exhausted unexpectedly")
    if {key: len(value) for key, value in roles.items()} != target_counts:
        raise RuntimeError("EXP-068 role counts do not match the frozen contract")

    file_by_name = {path.stem: path for path in files}
    payload = {
        "experiment": "EXP-068",
        "protocol_revision": config["protocol_revision"],
        "assignment": config["data"]["role_assignment"],
        "selection_inputs": "filename_only",
        "npz_content_accessed": False,
        "model_accessed": False,
        "exposed_external_sequences": sorted(exposed),
        "roles": {
            role: [
                {
                    "sequence": name,
                    "relative_path": str(file_by_name[name].relative_to(root)),
                    "bytes": file_by_name[name].stat().st_size,
                    "filename_family": family(name),
                }
                for name in values
            ]
            for role, values in roles.items()
        },
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({role: len(rows) for role, rows in payload["roles"].items()}, indent=2))


if __name__ == "__main__":
    main()
