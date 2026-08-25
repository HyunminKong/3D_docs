#!/usr/bin/env python3
"""Metadata-only DL3DV revisit inventory and immutable scene split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _inventory_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,jk->nik", left.transpose(0, 2, 1), right)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _pairs(metadata_path: Path, root: Path, config: dict) -> tuple[dict, list[dict]]:
    metadata = json.loads(metadata_path.read_text())
    frames = metadata["frames"]
    transforms = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    centers = transforms[:, :3, 3]
    rotations = transforms[:, :3, :3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive = steps[steps > 1e-8]
    if not len(positive):
        return {"frames": len(frames), "median_step": None, "candidate_pairs": 0}, []
    median_step = float(np.median(positive))
    gap = int(config["revisit"]["minimum_temporal_gap_frames"])
    radius = float(config["revisit"]["maximum_translation_in_median_steps"])
    maximum_angle = float(config["revisit"]["maximum_rotation_degrees"])
    candidates = []
    scene = metadata_path.parent.parent.name

    def image_path(index: int) -> str:
        filename = Path(frames[index]["file_path"]).name
        return str(root / scene / "nerfstudio" / config["data"]["image_directory"] / filename)

    for target in range(max(gap, 1), len(frames)):
        source_indices = np.arange(1, target - gap + 1)
        if not len(source_indices):
            continue
        translation = np.linalg.norm(centers[source_indices] - centers[target], axis=1) / median_step
        within = source_indices[translation <= radius]
        if not len(within):
            continue
        angles = _rotation_angle_degrees(rotations[within], rotations[target])
        keep = angles <= maximum_angle
        within = within[keep]
        angles = angles[keep]
        if not len(within):
            continue
        translations = np.linalg.norm(centers[within] - centers[target], axis=1) / median_step
        order = np.lexsort((within, angles, translations))
        source = int(within[order[0]])
        candidates.append(
            {
                "scene": scene,
                "source_previous_index": source - 1,
                "source_index": source,
                "target_previous_index": target - 1,
                "target_index": target,
                "source_previous_rgb": image_path(source - 1),
                "source_rgb": image_path(source),
                "target_previous_rgb": image_path(target - 1),
                "target_rgb": image_path(target),
                "temporal_gap_frames": target - source,
                "translation_in_median_steps": float(translations[order[0]]),
                "rotation_degrees": float(angles[order[0]]),
            }
        )
    return {
        "frames": len(frames),
        "median_step": median_step,
        "candidate_pairs": len(candidates),
    }, candidates


def _subsample(rows: list[dict], maximum: int) -> list[dict]:
    if len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum).round().astype(np.int64)
    return [rows[int(index)] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-039_dl3dv_source_safe_partition_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    manifest_paths = {
        "train": Path(config["output"]["train_manifest"]),
        "validation": Path(config["output"]["validation_manifest"]),
        "terminal": Path(config["output"]["terminal_manifest"]),
    }
    if result_path.exists() or any(path.exists() for path in manifest_paths.values()):
        raise RuntimeError("EXP-039 output already exists")

    root = Path(config["data"]["root"])
    metadata_paths = sorted(root.glob(config["data"]["metadata_glob"]))
    inventory_hash = _inventory_digest(root, metadata_paths)
    if not (
        len(metadata_paths) == int(config["data"]["expected_scenes"])
        and inventory_hash == config["data"]["metadata_inventory_sha256"]
    ):
        raise RuntimeError("EXP-039 metadata inventory changed")

    scene_rows = []
    pair_rows = {}
    minimum_pairs = int(config["revisit"]["minimum_pairs_per_scene"])
    maximum_pairs = int(config["revisit"]["maximum_pairs_per_scene"])
    for path in metadata_paths:
        audit, pairs = _pairs(path, root, config)
        scene = path.parent.parent.name
        selected = _subsample(pairs, maximum_pairs) if len(pairs) >= minimum_pairs else []
        paths_exist = all(
            Path(row[key]).is_file()
            for row in selected
            for key in ("source_previous_rgb", "source_rgb", "target_previous_rgb", "target_rgb")
        )
        scene_rows.append({"scene": scene, **audit, "selected_pairs": len(selected), "paths_exist": paths_exist})
        if selected:
            pair_rows[scene] = selected

    eligible = sorted(
        pair_rows,
        key=lambda scene: hashlib.sha256(f"{scene}:{config['seed']}".encode()).hexdigest(),
    )
    train_end = int(config["split"]["train_scenes"])
    validation_end = train_end + int(config["split"]["validation_scenes"])
    terminal_end = validation_end + int(config["split"]["terminal_scenes"])
    assignments = {
        "train": eligible[:train_end],
        "validation": eligible[train_end:validation_end],
        "terminal": eligible[validation_end:terminal_end],
    }
    manifests = {
        role: [
            {**pair, "role": role, "pair_id": f"{scene}:{index:03d}"}
            for scene in scenes
            for index, pair in enumerate(pair_rows[scene])
        ]
        for role, scenes in assignments.items()
    }
    for role, path in manifest_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifests[role], indent=2, allow_nan=False))

    split_sets = [set(value) for value in assignments.values()]
    disjoint = all(split_sets[i].isdisjoint(split_sets[j]) for i in range(3) for j in range(i + 1, 3))
    all_paths_exist = all(row["paths_exist"] for row in scene_rows if row["selected_pairs"])
    checks = {
        "exact_metadata_inventory": len(metadata_paths) == int(config["data"]["expected_scenes"])
        and inventory_hash == config["data"]["metadata_inventory_sha256"],
        "exact_eligible_scenes": len(eligible) == int(config["success"]["exact_eligible_scenes"]),
        "exact_split_sizes": all(
            len(assignments[role]) == int(config["split"][f"{role}_scenes"])
            for role in assignments
        ),
        "minimum_pair_counts": len(manifests["train"]) >= int(config["success"]["minimum_train_pairs"])
        and len(manifests["validation"]) >= int(config["success"]["minimum_validation_pairs"])
        and len(manifests["terminal"]) >= int(config["success"]["minimum_terminal_pairs"]),
        "scene_disjoint": disjoint,
        "all_rgb_paths_exist": all_paths_exist,
        "no_sensor_decoded": True,
        "no_model_access": True,
    }
    result = {
        "experiment": "EXP-039",
        "stage": "dl3dv_metadata_source_safe_partition",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "metadata_inventory_sha256": inventory_hash,
        "metadata_scenes": len(metadata_paths),
        "eligible_scenes": len(eligible),
        "split": {
            role: {
                "scenes": len(assignments[role]),
                "pairs": len(manifests[role]),
                "scene_ids": assignments[role],
                "manifest": str(manifest_paths[role]),
                "manifest_sha256": _sha256(manifest_paths[role]),
            }
            for role in assignments
        },
        "scene_audit": scene_rows,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "image_decoded": False,
        "depth_or_geometry_label_accessed": False,
        "model_output_accessed": False,
        "terminal_manifest_locked_before_model_fit": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"eligible_scenes": len(eligible), "split": result["split"], "gate": result["registered_gate"]}, indent=2))


if __name__ == "__main__":
    main()
