#!/usr/bin/env python3
"""Metadata-only feasibility audit of untouched nuScenes official-test revisits."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree


def _rotation(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _components(nodes: set[str], edges: list[dict]) -> list[set[str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in edges:
        left, right = find(edge["left"]), find(edge["right"])
        if left != right:
            parent[left] = right
    groups: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        groups[find(node)].add(node)
    return list(groups.values())


def _previous_manifest_scenes(pattern: str) -> set[str]:
    names: set[str] = set()
    for filename in glob.glob(pattern):
        try:
            payload = json.loads(Path(filename).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            for key in ("source_scene", "target_scene", "scene"):
                value = row.get(key)
                if isinstance(value, str) and value.startswith("scene-"):
                    names.add(value)
            for segment in ("a", "b", "a_prime"):
                value = row.get(segment)
                if isinstance(value, dict) and isinstance(value.get("scene"), str):
                    names.add(value["scene"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-021_independent_test_inventory_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output_cache = Path(config["output"]["inventory_cache"])
    output_summary = Path(config["output"]["summary"])
    if output_cache.exists() or output_summary.exists():
        raise RuntimeError("EXP-021 inventory output already exists")

    root = Path(config["metadata"]["root"])
    base = root / config["metadata"]["version"]
    reference = root / config["metadata"]["reference_version"]
    scenes = json.loads((base / "scene.json").read_text())
    samples = json.loads((base / "sample.json").read_text())
    sample_data = json.loads((base / "sample_data.json").read_text())
    sensors = {row["token"]: row for row in json.loads((base / "sensor.json").read_text())}
    calibrations = {
        row["token"]: row
        for row in json.loads((base / "calibrated_sensor.json").read_text())
    }
    ego = {row["token"]: row for row in json.loads((base / "ego_pose.json").read_text())}
    logs = {row["token"]: row for row in json.loads((base / "log.json").read_text())}

    channels = {
        token: sensors[row["sensor_token"]]["channel"]
        for token, row in calibrations.items()
    }
    camera_channel = config["metadata"]["camera_channel"]
    lidar_channel = config["metadata"]["lidar_channel"]
    camera_calibrations = {
        token: row for token, row in calibrations.items() if channels[token] == camera_channel
    }
    sample_to_scene = {row["token"]: row["scene_token"] for row in samples}
    by_scene_camera: dict[str, list[dict]] = defaultdict(list)
    keyframe_files = {camera_channel: [], lidar_channel: []}
    for row in sample_data:
        channel = channels[row["calibrated_sensor_token"]]
        if row["is_key_frame"] and channel in keyframe_files:
            keyframe_files[channel].append(root / row["filename"])
        if row["calibrated_sensor_token"] in camera_calibrations:
            scene_token = sample_to_scene.get(row["sample_token"])
            if scene_token is not None:
                by_scene_camera[scene_token].append(row)

    scene_meta = {row["token"]: row for row in scenes}
    minimum_frames = int(config["metadata"]["minimum_camera_frames"])
    scene_info = {}
    for token, rows in by_scene_camera.items():
        if len(rows) < minimum_frames:
            continue
        meta = scene_meta[token]
        ordered = sorted(rows, key=lambda row: row["timestamp"])
        centers = []
        for row in ordered:
            calibration = camera_calibrations[row["calibrated_sensor_token"]]
            pose = ego[row["ego_pose_token"]]
            center = (
                _rotation(pose["rotation"])
                @ np.asarray(calibration["translation"], dtype=np.float64)
                + np.asarray(pose["translation"], dtype=np.float64)
            )
            centers.append(center)
        scene_info[meta["name"]] = {
            "scene_token": token,
            "location": logs[meta["log_token"]]["location"],
            "frames": len(ordered),
            "timestamps": [int(row["timestamp"]) for row in ordered],
            "sample_data_tokens": [row["token"] for row in ordered],
            "centers": np.asarray(centers, dtype=np.float64),
        }

    threshold = float(config["overlap"]["maximum_distance_m"])
    by_location: dict[str, list[str]] = defaultdict(list)
    for name, row in scene_info.items():
        by_location[row["location"]].append(name)
    edges = []
    for location, location_scenes in sorted(by_location.items()):
        names = sorted(location_scenes)
        trees = {name: cKDTree(scene_info[name]["centers"]) for name in names}
        bounds = {
            name: (scene_info[name]["centers"].min(0), scene_info[name]["centers"].max(0))
            for name in names
        }
        for left_index, left in enumerate(names):
            left_min, left_max = bounds[left]
            for right in names[left_index + 1:]:
                right_min, right_max = bounds[right]
                gap = np.maximum(
                    0.0, np.maximum(left_min, right_min) - np.minimum(left_max, right_max)
                )
                if float(np.linalg.norm(gap)) > threshold:
                    continue
                distances, indices = trees[right].query(scene_info[left]["centers"], k=1)
                left_anchor = int(np.argmin(distances))
                minimum = float(distances[left_anchor])
                if minimum <= threshold:
                    edges.append({
                        "left": left,
                        "right": right,
                        "location": location,
                        "minimum_distance_m": minimum,
                        "left_anchor": left_anchor,
                        "right_anchor": int(indices[left_anchor]),
                    })

    edge_nodes = {name for edge in edges for name in (edge["left"], edge["right"])}
    groups = _components(edge_nodes, edges)
    component_of = {name: index for index, group in enumerate(groups) for name in group}
    components = []
    for index, group in enumerate(groups):
        group_edges = [edge for edge in edges if edge["left"] in group]
        locations = {scene_info[name]["location"] for name in group}
        if len(locations) != 1:
            raise RuntimeError("overlap component crosses locations")
        components.append({
            "component_id": index,
            "location": next(iter(locations)),
            "scenes": sorted(group),
            "scene_count": len(group),
            "undirected_edges": len(group_edges),
        })
    for edge in edges:
        edge["component_id"] = component_of[edge["left"]]

    reference_tokens = {
        row["token"] for row in json.loads((reference / "scene.json").read_text())
    }
    test_tokens = {row["token"] for row in scenes}
    previous_names = _previous_manifest_scenes(config["metadata"]["previous_manifest_glob"])
    eligible_names = set(scene_info)
    file_existence = {
        channel: {
            "count": len(files),
            "existing": sum(path.is_file() for path in files),
            "fraction": float(np.mean([path.is_file() for path in files])) if files else 0.0,
        }
        for channel, files in keyframe_files.items()
    }

    success = config["success"]
    checks = {
        "eligible_scenes": len(scene_info) >= int(success["minimum_eligible_scenes"]),
        "overlap_scenes": len(edge_nodes) >= int(success["minimum_overlap_scenes"]),
        "undirected_edges": len(edges) >= int(success["minimum_undirected_edges"]),
        "components": len(components) >= int(success["minimum_components"]),
        "locations": len({row["location"] for row in scene_info.values()}) >= int(success["minimum_locations"]),
        "reference_scene_token_disjoint": not (test_tokens & reference_tokens),
        "previous_manifest_name_disjoint": not (eligible_names & previous_names),
        "camera_files": file_existence[camera_channel]["fraction"] >= float(success["minimum_keyframe_file_existence"]),
        "lidar_files": file_existence[lidar_channel]["fraction"] >= float(success["minimum_keyframe_file_existence"]),
    }
    inventory = {
        "experiment": "EXP-021",
        "stage": "independent_test_metadata_inventory",
        "protocol_revision": config["protocol_revision"],
        "metadata_only": True,
        "image_decoded": False,
        "model_output_accessed": False,
        "lidar_decoded": False,
        "scenes": {
            name: {key: value for key, value in row.items() if key != "centers"}
            for name, row in scene_info.items()
        },
        "edges": edges,
        "components": components,
    }
    summary = {
        "experiment": "EXP-021",
        "stage": "independent_test_metadata_inventory",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "metadata_only": True,
        "image_decoded": False,
        "model_output_accessed": False,
        "lidar_decoded": False,
        "total_scenes": len(scenes),
        "eligible_scenes": len(scene_info),
        "eligible_by_location": dict(Counter(row["location"] for row in scene_info.values())),
        "overlap_scenes": len(edge_nodes),
        "undirected_overlap_edges": len(edges),
        "connected_components": len(components),
        "component_scene_sizes": sorted((len(group) for group in groups), reverse=True),
        "component_edge_sizes": sorted((row["undirected_edges"] for row in components), reverse=True),
        "reference_scene_token_overlap": len(test_tokens & reference_tokens),
        "previous_manifest_name_overlap": len(eligible_names & previous_names),
        "keyframe_file_existence": file_existence,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "inventory_cache": str(output_cache),
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_cache.write_text(json.dumps(inventory, indent=2, allow_nan=False))
    output_summary.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
