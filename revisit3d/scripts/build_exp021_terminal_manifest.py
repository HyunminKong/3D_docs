#!/usr/bin/env python3
"""Freeze the untouched EXP-021 official-test terminal revisit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
    radius = max(
        context + queries,
        int(frames * float(config["segments"]["local_radius_fraction"])),
    )
    start = max(0, min(anchor - radius // 2, frames - radius))
    return _segment(scene, start, start + radius, context, queries)


def _distant(scene: str, anchor: int, frames: int, config: dict):
    context = int(config["segments"]["context_frames"])
    queries = int(config["segments"]["query_frames"])
    radius = max(
        context + queries,
        int(frames * float(config["segments"]["local_radius_fraction"])),
    )
    start = 0 if anchor >= frames // 2 else frames - radius
    return _segment(scene, start, start + radius, context, queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-021_terminal_manifest_v11.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    inventory_path = Path(config["source"]["inventory"])
    summary_path = Path(config["source"]["inventory_summary"])
    manifest_path = Path(config["output"]["manifest"])
    result_path = Path(config["output"]["result"])
    if manifest_path.exists() or result_path.exists():
        raise RuntimeError("EXP-021 terminal manifest output already exists")

    inventory = json.loads(inventory_path.read_text())
    summary = json.loads(summary_path.read_text())
    if not (
        inventory.get("metadata_only") is True
        and inventory.get("image_decoded") is False
        and inventory.get("model_output_accessed") is False
        and inventory.get("lidar_decoded") is False
        and summary["registered_gate"]["passed"] is True
    ):
        raise RuntimeError("EXP-021 v1.1 requires the passing untouched metadata audit")

    scenes = inventory["scenes"]
    episodes = []
    for edge in inventory["edges"]:
        directions = (
            (edge["left"], edge["right"], edge["left_anchor"], edge["right_anchor"]),
            (edge["right"], edge["left"], edge["right_anchor"], edge["left_anchor"]),
        )
        for source, target, source_anchor, target_anchor in directions:
            source_frames = int(scenes[source]["frames"])
            target_frames = int(scenes[target]["frames"])
            episodes.append({
                "episode_id": f"{source}__{target}",
                "split": "terminal_test",
                "component_id": int(edge["component_id"]),
                "source_scene": source,
                "target_scene": target,
                "location": edge["location"],
                "min_overlap_m": float(edge["minimum_distance_m"]),
                "a": asdict(_local(source, int(source_anchor), source_frames, config)),
                "b": asdict(_distant(source, int(source_anchor), source_frames, config)),
                "a_prime": asdict(_local(target, int(target_anchor), target_frames, config)),
            })
    episodes.sort(key=lambda row: row["episode_id"])
    if len({row["episode_id"] for row in episodes}) != len(episodes):
        raise RuntimeError("duplicate directional episode ID")

    used_scenes = set()
    used_components = set()
    used_locations = set()
    for row in episodes:
        used_scenes.update((row["source_scene"], row["target_scene"]))
        used_components.add(row["component_id"])
        used_locations.add(row["location"])
        for key in ("a", "b", "a_prime"):
            segment = row[key]
            context = set(segment["frames"])
            query = set(segment["query_frames"])
            if context & query:
                raise RuntimeError("context/query frame overlap")
            indices = segment["frames"] + segment["query_frames"]
            if min(indices) < 0 or max(indices) >= scenes[segment["scene"]]["frames"]:
                raise RuntimeError("segment frame index outside scene")

    health = config["minimum_health"]
    checks = {
        "directional_episodes": len(episodes) >= int(health["directional_episodes"]),
        "scenes": len(used_scenes) >= int(health["scenes"]),
        "components": len(used_components) >= int(health["components"]),
        "locations": len(used_locations) >= int(health["locations"]),
        "single_terminal_split": {row["split"] for row in episodes} == {"terminal_test"},
    }
    if not all(checks.values()):
        raise RuntimeError(f"EXP-021 terminal-manifest health failed: {checks}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(episodes, indent=2, allow_nan=False))
    result = {
        "experiment": "EXP-021",
        "stage": "terminal_manifest_freeze",
        "protocol_revision": config["protocol_revision"],
        "metadata_only": True,
        "image_decoded": False,
        "model_output_accessed": False,
        "lidar_decoded": False,
        "config": str(config_path),
        "inventory": str(inventory_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "directional_episodes": len(episodes),
        "scenes": len(used_scenes),
        "components": len(used_components),
        "locations": sorted(used_locations),
        "episodes_by_location": dict(Counter(row["location"] for row in episodes)),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "terminal_test_locked": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
