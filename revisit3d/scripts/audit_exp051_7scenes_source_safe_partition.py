#!/usr/bin/env python3
"""Metadata-only 7Scenes partition freeze for EXP-051."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


FRAME_RE = re.compile(r"frame-(\d{6})\.(color\.png|depth\.png|pose\.txt)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_inventory(root: Path, sequence: Path) -> dict:
    modalities: dict[str, set[str]] = {
        "color.png": set(),
        "depth.png": set(),
        "pose.txt": set(),
    }
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(sequence.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            continue
        frame, modality = match.groups()
        modalities[modality].add(frame)
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
    common = set.intersection(*modalities.values())
    union = set.union(*modalities.values())
    return {
        "scene": sequence.parent.name,
        "sequence": sequence.name,
        "relative_path": sequence.relative_to(root).as_posix(),
        "frames": len(common),
        "complete_triplets": len(common) == len(union)
        and all(len(values) == len(common) for values in modalities.values()),
        "modality_counts": {key: len(value) for key, value in modalities.items()},
        "total_bytes": total_bytes,
        "path_size_inventory_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-051_ttt3r_metric_aligned_prerequisites_v10.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    output = config["output"]
    root = Path(data["root"])

    scenes = sorted(path.name for path in root.iterdir() if path.is_dir())
    ordered = sorted(
        scenes,
        key=lambda scene: hashlib.sha256(
            f"{config['seed']}:{scene}".encode()
        ).hexdigest(),
    )
    n_train = int(data["train_scenes"])
    n_validation = int(data["validation_scenes"])
    assignments = {
        "train": ordered[:n_train],
        "validation": ordered[n_train : n_train + n_validation],
        "terminal": ordered[n_train + n_validation :],
    }
    if set.union(*(set(value) for value in assignments.values())) != set(scenes):
        raise RuntimeError("EXP-051 scene assignment is incomplete")
    if sum(len(value) for value in assignments.values()) != sum(
        len(set(value)) for value in assignments.values()
    ):
        raise RuntimeError("EXP-051 roles overlap")

    role_manifests = {}
    for role, role_scenes in assignments.items():
        sequences = []
        for scene in role_scenes:
            for sequence in sorted(
                path for path in (root / scene).glob("seq-*") if path.is_dir()
            ):
                sequences.append(_sequence_inventory(root, sequence))
        role_manifests[role] = {
            "experiment": "EXP-051",
            "role": role,
            "root": str(root),
            "scenes": role_scenes,
            "sequences": sequences,
            "sensor_content_decoded": False,
            "model_accessed": False,
            "terminal_accessed": False,
        }

    all_sequences = [
        sequence
        for manifest in role_manifests.values()
        for sequence in manifest["sequences"]
    ]
    counts = {
        role: {
            "scenes": len(manifest["scenes"]),
            "sequences": len(manifest["sequences"]),
            "frames": sum(item["frames"] for item in manifest["sequences"]),
        }
        for role, manifest in role_manifests.items()
    }
    checks = {
        "exact_total_scenes": len(scenes) == int(data["expected_total_scenes"]),
        "exact_total_sequences": len(all_sequences)
        == int(data["expected_total_sequences"]),
        "exact_total_frames": sum(item["frames"] for item in all_sequences)
        == int(data["expected_total_frames"]),
        "exact_role_frames": counts["train"]["frames"]
        == int(data["expected_train_frames"])
        and counts["validation"]["frames"]
        == int(data["expected_validation_frames"])
        and counts["terminal"]["frames"]
        == int(data["expected_terminal_frames"]),
        "scene_disjoint": not (
            set(assignments["train"]) & set(assignments["validation"])
            or set(assignments["train"]) & set(assignments["terminal"])
            or set(assignments["validation"]) & set(assignments["terminal"])
        ),
        "complete_triplets": all(item["complete_triplets"] for item in all_sequences),
        "inventory_metadata_only": True,
    }
    result_path = Path(output["inventory"])
    manifest_paths = {
        role: Path(output[f"{role}_manifest"])
        for role in ("train", "validation", "terminal")
    }
    if result_path.exists() or any(path.exists() for path in manifest_paths.values()):
        raise RuntimeError("EXP-051 metadata artifact already exists")
    for role, path in manifest_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(role_manifests[role], indent=2, allow_nan=False))

    result = {
        "experiment": "EXP-051",
        "stage": "metadata_only_source_safe_partition",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "scene_order": ordered,
        "assignments": assignments,
        "counts": counts,
        "manifest_sha256": {
            role: _sha256(path) for role, path in manifest_paths.items()
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "sensor_content_decoded": False,
        "model_accessed": False,
        "terminal_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
