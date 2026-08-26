#!/usr/bin/env python3
"""Build the metadata-only matched low-parallax premise for EXP-049."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _motion(rows: list[dict], template: str) -> list[dict]:
    enriched = []
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    for scene, scene_rows in by_scene.items():
        metadata_path = Path(template.format(scene=scene))
        metadata = json.loads(metadata_path.read_text())
        transforms = np.asarray(
            [frame["transform_matrix"] for frame in metadata["frames"]],
            dtype=np.float64,
        )
        centers = transforms[:, :3, 3]
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        positive = steps[steps > 1e-8]
        if not len(positive):
            raise RuntimeError(f"EXP-049 scene has no nonzero camera step: {scene}")
        median = float(np.median(positive))
        for row in scene_rows:
            source = int(row["source_index"])
            target = int(row["target_index"])
            enriched.append(
                {
                    **row,
                    "source_adjacent_translation_in_median_steps": float(
                        np.linalg.norm(centers[source] - centers[source - 1]) / median
                    ),
                    "target_adjacent_translation_in_median_steps": float(
                        np.linalg.norm(centers[target] - centers[target - 1]) / median
                    ),
                    "motion_metadata": str(metadata_path),
                }
            )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-049_low_parallax_oracle_manifest_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    manifest_path = Path(config["output"]["manifest"])
    audit_path = Path(config["output"]["audit"])
    if manifest_path.exists() or audit_path.exists():
        raise RuntimeError("EXP-049 metadata output already exists")
    source_path = Path(config["data"]["source_manifest"])
    if not (
        _sha256(source_path) == config["data"]["source_manifest_sha256"]
        and config["data"]["role"] == "train"
        and config["data"]["validation_access"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-049 source-safe metadata contract failed")
    source_rows = json.loads(source_path.read_text())
    if not all(row["role"] == "train" for row in source_rows):
        raise RuntimeError("EXP-049 source manifest is not train-only")
    rows = _motion(source_rows, config["data"]["metadata_template"])
    source_minimum = float(config["definition"]["minimum_source_adjacent_translation"])
    low_maximum = float(
        config["definition"]["maximum_low_parallax_target_translation"]
    )
    sufficient_minimum = float(
        config["definition"]["minimum_sufficient_target_translation"]
    )
    selected = []
    shared_scenes = []
    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = sorted(
            [row for row in rows if row["scene"] == scene],
            key=lambda row: row["pair_id"],
        )
        low = [
            row
            for row in scene_rows
            if row["source_adjacent_translation_in_median_steps"] >= source_minimum
            and row["target_adjacent_translation_in_median_steps"] <= low_maximum
        ]
        sufficient = [
            row
            for row in scene_rows
            if row["source_adjacent_translation_in_median_steps"] >= source_minimum
            and row["target_adjacent_translation_in_median_steps"]
            >= sufficient_minimum
        ]
        if not low or not sufficient:
            continue
        shared_scenes.append(scene)
        selected.extend(
            [
                {**low[0], "information_regime": "low_parallax_complementary"},
                {**sufficient[0], "information_regime": "motion_sufficient"},
            ]
        )
    regime_counts = {
        regime: sum(row["information_regime"] == regime for row in selected)
        for regime in ("low_parallax_complementary", "motion_sufficient")
    }
    paths_exist = all(
        Path(row[key]).is_file()
        for row in selected
        for key in (
            "source_previous_rgb",
            "source_rgb",
            "target_previous_rgb",
            "target_rgb",
        )
    )
    checks = {
        "exact_shared_scenes": len(shared_scenes)
        == int(config["success"]["exact_shared_scenes"]),
        "exact_pairs": len(selected) == int(config["success"]["exact_pairs"]),
        "exact_pairs_per_regime": all(
            count == int(config["success"]["exact_pairs_per_regime"])
            for count in regime_counts.values()
        ),
        "all_rgb_paths": paths_exist,
        "metadata_only": True,
        "train_only": all(row["role"] == "train" for row in selected),
    }
    if not all(checks.values()):
        raise RuntimeError(f"EXP-049 metadata gate failed: {checks}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(selected, indent=2, allow_nan=False))
    audit = {
        "experiment": "EXP-049",
        "stage": "metadata_only_low_parallax_manifest",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "source_rows": len(source_rows),
        "shared_scenes": len(shared_scenes),
        "pairs": len(selected),
        "regime_counts": regime_counts,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "image_decoded": False,
        "model_accessed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
