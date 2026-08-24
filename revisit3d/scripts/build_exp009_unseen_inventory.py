#!/usr/bin/env python3
"""Build a metadata-only overlap inventory from nuScenes scenes unseen in EXP-001--008."""

from __future__ import annotations

import argparse
import glob
import hashlib
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


def _blacklist(config: dict) -> tuple[set[str], dict]:
    scenes: set[str] = set()
    root_counts = {}
    for root_name in config["blacklist"]["converted_roots"]:
        root = Path(root_name)
        names = {
            path.name for path in root.iterdir()
            if path.is_dir() or path.is_symlink()
        } if root.exists() else set()
        scenes.update(names)
        root_counts[root_name] = len(names)
    manifest_scenes = set()
    for name in glob.glob(config["blacklist"]["manifest_glob"]):
        try:
            payload = json.loads(Path(name).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            for key in ("source_scene", "target_scene", "scene"):
                if isinstance(row.get(key), str) and row[key].startswith("scene-"):
                    manifest_scenes.add(row[key])
            for segment in ("a", "b", "a_prime"):
                if isinstance(row.get(segment), dict) and isinstance(row[segment].get("scene"), str):
                    manifest_scenes.add(row[segment]["scene"])
    scenes.update(manifest_scenes)
    return scenes, {"converted_root_counts": root_counts, "manifest_scenes": len(manifest_scenes)}


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
    groups = defaultdict(set)
    for node in nodes:
        groups[find(node)].add(node)
    return list(groups.values())


def _assign_components(component_rows: list[dict], ratios: dict[str, float], seed: int) -> None:
    split_order = ("train", "val", "test")
    by_location = defaultdict(list)
    for row in component_rows:
        by_location[row["location"]].append(row)
    for location, rows in sorted(by_location.items()):
        total = sum(row["undirected_edges"] for row in rows)
        targets = {name: max(ratios[name] * total, 1e-9) for name in split_order}
        loads = {name: 0 for name in split_order}
        ordered = sorted(rows, key=lambda row: (
            -row["undirected_edges"],
            hashlib.sha1(f"{seed}:{row['component_id']}".encode()).hexdigest(),
        ))
        for row in ordered:
            split = min(split_order, key=lambda name: (loads[name] / targets[name], split_order.index(name)))
            row["split"] = split
            loads[split] += row["undirected_edges"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_unseen_benchmark_inventory_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    cache_path = Path(config["output"]["inventory_cache"])
    summary_path = Path(config["output"]["summary"])
    if cache_path.exists() or summary_path.exists():
        raise RuntimeError("EXP-009 inventory output already exists")

    blacklist, blacklist_audit = _blacklist(config)
    base = Path(config["metadata"]["root"]) / config["metadata"]["version"]
    scenes = json.loads((base / "scene.json").read_text())
    samples = json.loads((base / "sample.json").read_text())
    sensors = {row["token"]: row for row in json.loads((base / "sensor.json").read_text())}
    calibrations = {
        row["token"]: row for row in json.loads((base / "calibrated_sensor.json").read_text())
    }
    ego = {row["token"]: row for row in json.loads((base / "ego_pose.json").read_text())}
    logs = {row["token"]: row for row in json.loads((base / "log.json").read_text())}
    sample_data = json.loads((base / "sample_data.json").read_text())

    channel = config["metadata"]["channel"]
    camera_calibrations = {
        token: row for token, row in calibrations.items()
        if sensors[row["sensor_token"]]["channel"] == channel
    }
    sample_to_scene = {row["token"]: row["scene_token"] for row in samples}
    by_scene_token = defaultdict(list)
    for row in sample_data:
        if row["calibrated_sensor_token"] in camera_calibrations:
            scene_token = sample_to_scene.get(row["sample_token"])
            if scene_token is not None:
                by_scene_token[scene_token].append(row)

    minimum_frames = int(config["metadata"]["minimum_frames"])
    scene_info = {}
    scene_meta = {row["token"]: row for row in scenes}
    for token, rows in by_scene_token.items():
        meta = scene_meta[token]
        name = meta["name"]
        if name in blacklist or len(rows) < minimum_frames:
            continue
        ordered = sorted(rows, key=lambda row: row["timestamp"])
        centers = []
        for row in ordered:
            calibration = camera_calibrations[row["calibrated_sensor_token"]]
            pose = ego[row["ego_pose_token"]]
            center = (
                _rotation(pose["rotation"]) @ np.asarray(calibration["translation"], dtype=np.float64)
                + np.asarray(pose["translation"], dtype=np.float64)
            )
            centers.append(center)
        scene_info[name] = {
            "scene_token": token,
            "location": logs[meta["log_token"]]["location"],
            "frames": len(ordered),
            "timestamps": [int(row["timestamp"]) for row in ordered],
            "sample_data_tokens": [row["token"] for row in ordered],
            "centers": np.asarray(centers, dtype=np.float64),
        }

    threshold = float(config["overlap"]["maximum_distance_m"])
    edges = []
    by_location = defaultdict(list)
    for name, row in scene_info.items():
        by_location[row["location"]].append(name)
    for location, names in sorted(by_location.items()):
        names = sorted(names)
        trees = {name: cKDTree(scene_info[name]["centers"]) for name in names}
        bounds = {
            name: (scene_info[name]["centers"].min(0), scene_info[name]["centers"].max(0))
            for name in names
        }
        for left_index, left in enumerate(names):
            left_min, left_max = bounds[left]
            for right in names[left_index + 1:]:
                right_min, right_max = bounds[right]
                gap = np.maximum(0.0, np.maximum(left_min, right_min) - np.minimum(left_max, right_max))
                if float(np.linalg.norm(gap)) > threshold:
                    continue
                distances, indices = trees[right].query(scene_info[left]["centers"], k=1)
                source_anchor = int(np.argmin(distances))
                minimum = float(distances[source_anchor])
                if minimum <= threshold:
                    edges.append({
                        "left": left, "right": right, "location": location,
                        "minimum_distance_m": minimum,
                        "left_anchor": source_anchor,
                        "right_anchor": int(indices[source_anchor]),
                    })
        print(json.dumps({"location": location, "scenes": len(names), "edges_so_far": len(edges)}), flush=True)

    edge_nodes = {name for edge in edges for name in (edge["left"], edge["right"])}
    components = _components(edge_nodes, edges)
    component_of = {name: index for index, group in enumerate(components) for name in group}
    component_rows = []
    for index, group in enumerate(components):
        group_edges = [edge for edge in edges if edge["left"] in group]
        locations = {scene_info[name]["location"] for name in group}
        if len(locations) != 1:
            raise RuntimeError("an overlap component crosses nuScenes locations")
        component_rows.append({
            "component_id": index,
            "location": next(iter(locations)),
            "scenes": sorted(group),
            "scene_count": len(group),
            "undirected_edges": len(group_edges),
        })
    _assign_components(component_rows, config["split"]["ratios"], int(config["seed"]))
    split_by_component = {row["component_id"]: row["split"] for row in component_rows}
    for edge in edges:
        edge["component_id"] = component_of[edge["left"]]
        edge["split"] = split_by_component[edge["component_id"]]

    cache_payload = {
        "experiment": "EXP-009", "protocol_revision": config["protocol_revision"],
        "metadata_only": True, "image_accessed": False, "model_output_accessed": False,
        "blacklisted_scenes": sorted(blacklist),
        "scenes": {
            name: {
                key: value for key, value in row.items() if key != "centers"
            } for name, row in scene_info.items()
        },
        "edges": edges, "components": component_rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload, indent=2, allow_nan=False))

    edge_split = Counter(edge["split"] for edge in edges)
    scene_split = Counter()
    for component in component_rows:
        scene_split[component["split"]] += component["scene_count"]
    summary = {
        "experiment": "EXP-009", "stage": "stage0_unseen_overlap_inventory",
        "protocol_revision": config["protocol_revision"],
        "metadata_only": True, "image_accessed": False, "model_output_accessed": False,
        "config": str(config_path),
        "blacklist_audit": {**blacklist_audit, "unique_blacklisted_scenes": len(blacklist)},
        "total_nuscenes_scenes": len(scenes),
        "eligible_unseen_scenes": len(scene_info),
        "eligible_by_location": dict(Counter(row["location"] for row in scene_info.values())),
        "overlap_scenes": len(edge_nodes), "undirected_overlap_edges": len(edges),
        "connected_components": len(component_rows),
        "component_scene_sizes": sorted([row["scene_count"] for row in component_rows], reverse=True),
        "component_edge_sizes": sorted([row["undirected_edges"] for row in component_rows], reverse=True),
        "split_undirected_edges": dict(edge_split),
        "split_directional_episodes": {key: 2 * value for key, value in edge_split.items()},
        "split_scenes": dict(scene_split),
        "cache": str(cache_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
