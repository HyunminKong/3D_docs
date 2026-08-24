#!/usr/bin/env python3
"""Expand EXP-006 train revisits while preserving every existing holdout.

The script may inspect scene poses and split membership to construct disjoint
physical-overlap components.  It never reads model outputs or held-out images.
Original validation/test episodes are copied unchanged; any enlarged component
touching either split is excluded from the expanded training set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from revisit3d.data import build_manifest


def _components(nodes: set[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root
    groups: dict[str, set[str]] = {}
    for node in nodes:
        groups.setdefault(find(node), set()).add(node)
    return list(groups.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots", nargs="+",
        default=[
            "tttLRM/data_example/nuscenes_processed",
            "tttLRM/data_example/nuscenes_prereg",
        ],
    )
    parser.add_argument(
        "--union-root", default="tttLRM/data_example/nuscenes_exp006_all",
        help="Generated symlink union; external data and excluded from Git.",
    )
    parser.add_argument("--nuscenes-meta", default="/mnt/ssd/nuscenes/v1.0-trainval")
    parser.add_argument(
        "--protected-manifest", default="revisit3d/manifests/nuscenes_revisit_dev.json",
    )
    parser.add_argument(
        "--selection-out", default="revisit3d/manifests/nuscenes_exp006_all_locations.json",
    )
    parser.add_argument(
        "--out", default="revisit3d/manifests/nuscenes_revisit_expanded_v27.json",
    )
    parser.add_argument(
        "--result", default="revisit3d/results/EXP-006/benchmark_expansion_train_v27.json",
    )
    parser.add_argument("--overlap-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=600)
    args = parser.parse_args()

    scene_paths: dict[str, Path] = {}
    for root_name in args.roots:
        root = Path(root_name)
        for camera in sorted(root.glob("*/opencv_cameras.json")):
            scene = camera.parent.name
            previous = scene_paths.setdefault(scene, camera.parent.resolve())
            if previous != camera.parent.resolve():
                raise RuntimeError(f"scene {scene} exists in multiple roots")
    if not scene_paths:
        raise RuntimeError("no converted scenes found")

    union_root = Path(args.union_root)
    union_root.mkdir(parents=True, exist_ok=True)
    for scene, source in sorted(scene_paths.items()):
        target = union_root / scene
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"union target {target} resolves to an unexpected scene")
        else:
            target.symlink_to(source, target_is_directory=True)

    meta = Path(args.nuscenes_meta)
    logs = {row["token"]: row["location"] for row in json.loads((meta / "log.json").read_text())}
    locations = {
        row["name"]: logs[row["log_token"]]
        for row in json.loads((meta / "scene.json").read_text())
    }
    missing_locations = sorted(set(scene_paths) - set(locations))
    if missing_locations:
        raise RuntimeError(f"nuScenes location missing for scenes: {missing_locations}")
    selection = {
        "purpose": "EXP-006 v2.7 expanded train construction; location metadata only",
        "units": [
            {"scene": scene, "location": locations[scene]}
            for scene in sorted(scene_paths)
        ],
    }
    selection_out = Path(args.selection_out)
    selection_out.parent.mkdir(parents=True, exist_ok=True)
    selection_out.write_text(json.dumps(selection, indent=2))

    generated = build_manifest(
        union_root, selection=selection_out, overlap_m=args.overlap_m, seed=args.seed,
    )
    edges = {
        tuple(sorted((episode.source_scene, episode.target_scene)))
        for episode in generated
    }
    nodes = {scene for edge in edges for scene in edge}
    components = _components(nodes, edges)
    component_of = {scene: index for index, component in enumerate(components) for scene in component}

    protected = json.loads(Path(args.protected_manifest).read_text())
    protected_splits: dict[str, set[str]] = {}
    for episode in protected:
        for scene in (episode["source_scene"], episode["target_scene"]):
            protected_splits.setdefault(scene, set()).add(episode["split"])
    safe_components, excluded_components = [], []
    for index, component in enumerate(components):
        labels = sorted(set().union(*(protected_splits.get(scene, set()) for scene in component)))
        record = {"component": index, "scenes": sorted(component), "protected_splits": labels}
        if any(label in ("val", "test") for label in labels):
            excluded_components.append(record)
        else:
            safe_components.append(record)
    safe_ids = {row["component"] for row in safe_components}
    train = [
        replace(episode, split="train")
        for episode in generated
        if component_of[episode.source_scene] in safe_ids
    ]
    holdout = [episode for episode in protected if episode["split"] in ("val", "test")]
    output_rows = [asdict(episode) for episode in sorted(train, key=lambda row: row.episode_id)]
    output_rows.extend(holdout)

    train_scenes = {
        scene for episode in output_rows if episode["split"] == "train"
        for scene in (episode["source_scene"], episode["target_scene"])
    }
    holdout_scenes = {
        scene for episode in output_rows if episode["split"] in ("val", "test")
        for scene in (episode["source_scene"], episode["target_scene"])
    }
    leakage = sorted(train_scenes & holdout_scenes)
    if leakage:
        raise RuntimeError(f"train/holdout scene leakage: {leakage}")
    if holdout != [episode for episode in output_rows if episode["split"] in ("val", "test")]:
        raise RuntimeError("protected holdout episodes changed")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_rows, indent=2))
    result = {
        "experiment": "EXP-006",
        "stage": "benchmark_expansion",
        "protocol_revision": "v2.7",
        "model_output_accessed": False,
        "heldout_image_accessed": False,
        "protected_manifest": args.protected_manifest,
        "scene_roots": args.roots,
        "union_root": args.union_root,
        "selection": args.selection_out,
        "manifest": args.out,
        "overlap_m": args.overlap_m,
        "converted_scenes": len(scene_paths),
        "undirected_overlap_pairs": len(edges),
        "nontrivial_overlap_components": len(components),
        "safe_train_components": len(safe_components),
        "excluded_holdout_components": len(excluded_components),
        "train_directional_episodes": len(train),
        "validation_directional_episodes": sum(row["split"] == "val" for row in holdout),
        "test_directional_episodes": sum(row["split"] == "test" for row in holdout),
        "train_holdout_scene_intersection": leakage,
        "safe_component_records": safe_components,
        "excluded_component_records": excluded_components,
    }
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "manifest": str(output),
        "train_episodes": len(train),
        "safe_components": len(safe_components),
        "protected_val": result["validation_directional_episodes"],
        "protected_test": result["test_directional_episodes"],
    }))


if __name__ == "__main__":
    main()
