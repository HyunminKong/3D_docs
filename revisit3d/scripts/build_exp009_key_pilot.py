#!/usr/bin/env python3
"""Select a location-balanced, train-only key benchmark without opening image pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def _hash(seed: int, *parts: str) -> str:
    return hashlib.sha1(":".join((str(seed), *parts)).encode()).hexdigest()


def _bbox(scene_root: Path, scene: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((scene_root / scene / "opencv_cameras.json").read_text())
    centers = []
    for frame in payload["frames"]:
        w2c = np.asarray(frame["w2c"], dtype=np.float64)
        centers.append(-w2c[:3, :3].T @ w2c[:3, 3])
    matrix = np.stack(centers)
    return matrix.min(0), matrix.max(0)


def _bbox_gap(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> float:
    left_min, left_max = left
    right_min, right_max = right
    gap = np.maximum(0.0, np.maximum(left_min, right_min) - np.minimum(left_max, right_max))
    return float(np.linalg.norm(gap))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_key_pilot_v13.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["pair_manifest"])
    result_path = Path(config["output"]["result"])
    if output.exists() or result_path.exists():
        raise RuntimeError("EXP-009 key-pilot output already exists")
    manifest = json.loads(Path(config["source"]["manifest"]).read_text())
    inventory = json.loads(Path(config["source"]["inventory"]).read_text())
    directional = {row["episode_id"]: row for row in manifest}
    seed = int(config["seed"])
    maximum = int(config["sampling"]["maximum_positive_edges_per_location"])
    by_location = defaultdict(list)
    for edge in inventory["edges"]:
        if edge["split"] == config["sampling"]["split"]:
            by_location[edge["location"]].append(edge)

    selected = []
    for location, edges in sorted(by_location.items()):
        ordered = sorted(edges, key=lambda row: _hash(seed, row["left"], row["right"]))
        selected.extend(ordered[:maximum])
    selected_counts = Counter(edge["location"] for edge in selected)
    minimum = int(config["sampling"]["minimum_positive_edges_per_location"])
    if any(count < minimum for count in selected_counts.values()) or len(selected_counts) != 4:
        raise RuntimeError(f"location-balanced pilot health failed: {selected_counts}")

    positives = []
    endpoints_by_location = defaultdict(list)
    view_positions = [int(value) for value in config["sampling"]["context_view_indices"]]
    for index, edge in enumerate(selected):
        episode = directional[f"{edge['left']}__{edge['right']}"]
        if episode["split"] != "train" or episode["component_id"] != edge["component_id"]:
            raise RuntimeError("selected edge disagrees with frozen train manifest")
        left = {
            "scene": episode["a"]["scene"],
            "frames": [episode["a"]["frames"][position] for position in view_positions],
        }
        right = {
            "scene": episode["a_prime"]["scene"],
            "frames": [episode["a_prime"]["frames"][position] for position in view_positions],
        }
        record = {
            "pair_id": f"positive-{index:04d}", "label": 1,
            "location": edge["location"], "component_id": int(edge["component_id"]),
            "minimum_distance_m": float(edge["minimum_distance_m"]),
            "left": left, "right": right,
        }
        positives.append(record)
        endpoints_by_location[edge["location"]].extend((left, right))

    scene_root = Path(config["source"]["scene_root"])
    used_scenes = {part["scene"] for row in positives for part in (row["left"], row["right"])}
    bounds = {scene: _bbox(scene_root, scene) for scene in used_scenes}
    direct_edges = {
        tuple(sorted((edge["left"], edge["right"]))) for edge in inventory["edges"]
    }
    negatives = []
    used_pairs = set()
    gap_minimum = float(config["sampling"]["minimum_negative_bbox_distance_m"])
    for index, positive in enumerate(positives):
        left = positive["left"]
        candidates = sorted(
            endpoints_by_location[positive["location"]],
            key=lambda row: _hash(seed, positive["pair_id"], row["scene"], str(row["frames"])),
        )
        choice = None
        for candidate in candidates:
            identity = tuple(sorted((left["scene"], candidate["scene"])))
            detailed = (left["scene"], tuple(left["frames"]), candidate["scene"], tuple(candidate["frames"]))
            if (
                left["scene"] != candidate["scene"]
                and identity not in direct_edges
                and detailed not in used_pairs
                and _bbox_gap(bounds[left["scene"]], bounds[candidate["scene"]]) >= gap_minimum
            ):
                choice = candidate
                used_pairs.add(detailed)
                break
        if choice is None:
            raise RuntimeError(f"no strict negative found for {positive['pair_id']}")
        negatives.append({
            "pair_id": f"negative-{index:04d}", "label": 0,
            "location": positive["location"], "component_id": positive["component_id"],
            "minimum_bbox_distance_m": _bbox_gap(bounds[left["scene"]], bounds[choice["scene"]]),
            "left": left, "right": choice,
        })

    pairs = positives + negatives
    pairs.sort(key=lambda row: row["pair_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pairs, indent=2, allow_nan=False))
    result = {
        "experiment": "EXP-009", "stage": "stage3_train_key_pilot_selection",
        "protocol_revision": config["protocol_revision"], "split": "train",
        "metadata_only": True, "image_pixels_accessed": False, "model_output_accessed": False,
        "config": str(config_path), "pair_manifest": str(output),
        "positive_pairs": len(positives), "negative_pairs": len(negatives),
        "positive_by_location": dict(selected_counts),
        "unique_scenes": len(used_scenes),
        "minimum_negative_bbox_distance_m": min(row["minimum_bbox_distance_m"] for row in negatives),
        "validation_accessed": False, "test_accessed": False, "passed": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
