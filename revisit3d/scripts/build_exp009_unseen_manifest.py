#!/usr/bin/env python3
"""Freeze directional A/B/A-prime episodes from the metadata-only EXP-009 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml

from revisit3d.data.revisit_benchmark import _segment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local(scene: str, anchor: int, frames: int, config: dict):
    context = int(config["segments"]["context_frames"])
    queries = int(config["segments"]["query_frames"])
    radius = max(context + queries, int(frames * float(config["segments"]["local_radius_fraction"])))
    start = max(0, min(anchor - radius // 2, frames - radius))
    return _segment(scene, start, start + radius, context, queries)


def _distant(scene: str, anchor: int, frames: int, config: dict):
    context = int(config["segments"]["context_frames"])
    queries = int(config["segments"]["query_frames"])
    radius = max(context + queries, int(frames * float(config["segments"]["local_radius_fraction"])))
    start = 0 if anchor >= frames // 2 else frames - radius
    return _segment(scene, start, start + radius, context, queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_unseen_manifest_v11.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    inventory_path = Path(config["source"]["inventory"])
    summary_path = Path(config["source"]["inventory_summary"])
    manifest_path = Path(config["output"]["manifest"])
    result_path = Path(config["output"]["result"])
    if manifest_path.exists() or result_path.exists():
        raise RuntimeError("EXP-009 manifest output already exists")
    inventory = json.loads(inventory_path.read_text())
    inventory_summary = json.loads(summary_path.read_text())
    if not (
        inventory.get("metadata_only") is True
        and inventory.get("image_accessed") is False
        and inventory.get("model_output_accessed") is False
        and inventory.get("protocol_revision") == "v1.0"
    ):
        raise RuntimeError("EXP-009 v1.1 requires the locked metadata-only v1.0 inventory")

    scenes = inventory["scenes"]
    blacklisted = set(inventory["blacklisted_scenes"])
    episodes = []
    for edge in inventory["edges"]:
        for left, right, left_anchor, right_anchor in (
            (edge["left"], edge["right"], edge["left_anchor"], edge["right_anchor"]),
            (edge["right"], edge["left"], edge["right_anchor"], edge["left_anchor"]),
        ):
            if left in blacklisted or right in blacklisted:
                raise RuntimeError("blacklisted scene reached the frozen manifest")
            left_frames = int(scenes[left]["frames"])
            right_frames = int(scenes[right]["frames"])
            episodes.append({
                "episode_id": f"{left}__{right}",
                "split": edge["split"],
                "component_id": int(edge["component_id"]),
                "source_scene": left,
                "target_scene": right,
                "location": edge["location"],
                "min_overlap_m": float(edge["minimum_distance_m"]),
                "a": asdict(_local(left, int(left_anchor), left_frames, config)),
                "b": asdict(_distant(left, int(left_anchor), left_frames, config)),
                "a_prime": asdict(_local(right, int(right_anchor), right_frames, config)),
            })
    episodes.sort(key=lambda row: row["episode_id"])
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise RuntimeError("duplicate directional episode ID")

    scenes_by_split = defaultdict(set)
    locations_by_split = defaultdict(set)
    components_by_split = defaultdict(set)
    for row in episodes:
        scenes_by_split[row["split"]].update((row["source_scene"], row["target_scene"]))
        locations_by_split[row["split"]].add(row["location"])
        components_by_split[row["split"]].add(row["component_id"])
        for key in ("a", "b", "a_prime"):
            segment = row[key]
            if set(segment["frames"]) & set(segment["query_frames"]):
                raise RuntimeError("context/query frame overlap")
            if min(segment["frames"] + segment["query_frames"]) < 0:
                raise RuntimeError("negative frame index")
            if max(segment["frames"] + segment["query_frames"]) >= scenes[segment["scene"]]["frames"]:
                raise RuntimeError("frame index exceeds scene length")
    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left < right and scenes_by_split[left] & scenes_by_split[right]:
                raise RuntimeError(f"scene leakage between {left} and {right}")
            if left < right and components_by_split[left] & components_by_split[right]:
                raise RuntimeError(f"component leakage between {left} and {right}")

    split_episodes = Counter(row["split"] for row in episodes)
    health = config["minimum_split_health"]
    checks = {
        "train_episode_minimum": split_episodes["train"] >= int(health["train_directional_episodes"]),
        "val_episode_minimum": split_episodes["val"] >= int(health["val_directional_episodes"]),
        "test_episode_minimum": split_episodes["test"] >= int(health["test_directional_episodes"]),
        "val_scene_minimum": len(scenes_by_split["val"]) >= int(health["val_scenes"]),
        "test_scene_minimum": len(scenes_by_split["test"]) >= int(health["test_scenes"]),
        "val_location_minimum": len(locations_by_split["val"]) >= int(health["val_locations"]),
        "test_location_minimum": len(locations_by_split["test"]) >= int(health["test_locations"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"EXP-009 manifest health failed: {checks}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(episodes, indent=2, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage1_unseen_manifest_freeze",
        "protocol_revision": config["protocol_revision"],
        "metadata_only": True, "image_accessed": False, "model_output_accessed": False,
        "config": str(config_path), "inventory": str(inventory_path),
        "inventory_summary": str(summary_path), "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "blacklisted_scene_intersection": 0,
        "split_directional_episodes": dict(split_episodes),
        "split_scenes": {key: len(value) for key, value in scenes_by_split.items()},
        "split_components": {key: len(value) for key, value in components_by_split.items()},
        "split_locations": {key: sorted(value) for key, value in locations_by_split.items()},
        "scene_intersections": {
            "train_val": len(scenes_by_split["train"] & scenes_by_split["val"]),
            "train_test": len(scenes_by_split["train"] & scenes_by_split["test"]),
            "val_test": len(scenes_by_split["val"] & scenes_by_split["test"]),
        },
        "component_intersections": {
            "train_val": len(components_by_split["train"] & components_by_split["val"]),
            "train_test": len(components_by_split["train"] & components_by_split["test"]),
            "val_test": len(components_by_split["val"] & components_by_split["test"]),
        },
        "health_checks": checks, "passed": all(checks.values()),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
