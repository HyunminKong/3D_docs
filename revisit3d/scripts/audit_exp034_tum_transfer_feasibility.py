#!/usr/bin/env python3
"""Metadata-only construction audit for a frozen TUM transfer stream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


INTRINSICS = {
    "rgbd_dataset_freiburg1_desk": [517.3, 516.5, 318.6, 255.3],
    "rgbd_dataset_freiburg2_xyz": [520.9, 521.0, 325.1, 249.7],
    "rgbd_dataset_freiburg3_long_office_household": [535.4, 539.2, 320.1, 247.6],
}


def _read_index(path: Path, fields: int) -> list[tuple[float, object]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        payload: object
        if fields == 1:
            payload = values[1]
        else:
            payload = [float(value) for value in values[1 : fields + 1]]
        rows.append((float(values[0]), payload))
    return rows


def _nearest(timestamp: float, rows: list[tuple[float, object]], tolerance: float):
    times = np.asarray([row[0] for row in rows])
    index = int(np.abs(times - timestamp).argmin())
    if abs(float(times[index]) - timestamp) > tolerance:
        return None
    return rows[index]


def _rotation_angle_deg(left: list[float], right: list[float]) -> float:
    left_q = np.asarray(left, dtype=np.float64)
    right_q = np.asarray(right, dtype=np.float64)
    left_q /= np.linalg.norm(left_q)
    right_q /= np.linalg.norm(right_q)
    return float(
        np.degrees(2 * np.arccos(np.clip(abs(left_q @ right_q), -1.0, 1.0)))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-034_tum_transfer_feasibility_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    root = Path(config["data"]["root"])
    tolerance = float(config["association"]["maximum_time_difference_s"])
    stream = config["stream"]
    revisit = config["revisit"]
    events = []
    sequence_summary = {}
    referenced_files = []

    for sequence in config["data"]["sequences"]:
        sequence_root = root / sequence
        rgb = _read_index(sequence_root / "rgb.txt", 1)
        depth = _read_index(sequence_root / "depth.txt", 1)
        pose = _read_index(sequence_root / "groundtruth.txt", 7)
        associated = []
        for timestamp, rgb_path in rgb:
            depth_row = _nearest(timestamp, depth, tolerance)
            pose_row = _nearest(timestamp, pose, tolerance)
            if depth_row is None or pose_row is None:
                continue
            referenced_files.extend(
                [sequence_root / str(rgb_path), sequence_root / str(depth_row[1])]
            )
            associated.append(
                {
                    "timestamp": timestamp,
                    "rgb": str(sequence_root / str(rgb_path)),
                    "depth": str(sequence_root / str(depth_row[1])),
                    "pose_t_qxyzw": pose_row[1],
                }
            )

        context_views = int(stream["context_views"])
        context_stride = int(stream["context_frame_stride"])
        query_views = int(stream["query_views"])
        query_stride = int(stream["query_frame_stride"])
        history = (context_views - 1) * context_stride
        future = query_views * query_stride
        anchors = range(
            history,
            len(associated) - future,
            int(stream["anchor_stride_associated_frames"]),
        )
        sequence_events = []
        for anchor in anchors:
            context_indices = list(
                range(anchor - history, anchor + 1, context_stride)
            )
            query_indices = [anchor + query_stride * i for i in range(1, query_views + 1)]
            center = associated[anchor]
            prior_revisits = []
            for prior in sequence_events:
                delta_t = center["timestamp"] - prior["timestamp"]
                left = np.asarray(center["pose_t_qxyzw"][:3])
                right = np.asarray(prior["pose_t_qxyzw"][:3])
                distance = float(np.linalg.norm(left - right))
                angle = _rotation_angle_deg(
                    center["pose_t_qxyzw"][3:], prior["pose_t_qxyzw"][3:]
                )
                if (
                    delta_t >= float(revisit["minimum_time_separation_s"])
                    and distance <= float(revisit["maximum_translation_m"])
                    and angle <= float(revisit["maximum_rotation_deg"])
                ):
                    prior_revisits.append(prior["event_id"])
            event = {
                "event_id": f"{sequence}-{len(sequence_events):04d}",
                "sequence": sequence,
                "component": sequence,
                "timestamp": center["timestamp"],
                "intrinsics_fx_fy_cx_cy": INTRINSICS[sequence],
                "depth_scale": 5000.0,
                "context": [associated[index] for index in context_indices],
                "query": [associated[index] for index in query_indices],
                "prior_revisit_event_ids": prior_revisits,
                "is_revisit_target": bool(prior_revisits),
            }
            sequence_events.append(event)
            events.append(event)
        sequence_summary[sequence] = {
            "rgb_rows": len(rgb),
            "associated_rows": len(associated),
            "stream_contexts": len(sequence_events),
            "revisit_targets": sum(x["is_revisit_target"] for x in sequence_events),
        }

    manifest_path = Path(config["output"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(events, indent=2, allow_nan=False))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    counts = {
        "sequences": len(sequence_summary),
        "stream_contexts": len(events),
        "revisit_targets": sum(x["is_revisit_target"] for x in events),
        "referenced_sensor_files": len(referenced_files),
        "missing_referenced_sensor_files": sum(
            not path.is_file() for path in referenced_files
        ),
    }
    checks = {
        "minimum_sequences": counts["sequences"]
        >= int(config["success"]["minimum_sequences"]),
        "minimum_stream_contexts": counts["stream_contexts"]
        >= int(config["success"]["minimum_stream_contexts"]),
        "minimum_revisit_targets": counts["revisit_targets"]
        >= int(config["success"]["minimum_revisit_targets"]),
        "all_referenced_files_exist": counts["missing_referenced_sensor_files"] == 0,
        "sensor_decoded": False,
        "model_output_accessed": False,
    }
    result = {
        "experiment": "EXP-034",
        "stage": "tum_zero_shot_metadata_feasibility",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "counts": counts,
        "sequences": sequence_summary,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "image_decoded": False,
        "depth_decoded": False,
        "model_output_accessed": False,
    }
    result_path = Path(config["output"]["result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
