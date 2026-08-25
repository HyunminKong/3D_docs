#!/usr/bin/env python3
"""Matched causal CUT3R/TTT3R evaluation on the frozen EXP-035 TUM stream."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from revisit3d.scripts.evaluate_exp010_absolute_geometry import _depth_metrics
from revisit3d.scripts.evaluate_exp035_tum_zero_shot import _query_depth_gt
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


METRICS = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")


def _views(images: list[dict], updates: list[bool]) -> list[dict]:
    views = []
    for index, (image, update) in enumerate(zip(images, updates)):
        tensor = image["img"]
        views.append(
            {
                "img": tensor,
                "ray_map": torch.full(
                    (tensor.shape[0], 6, tensor.shape[-2], tensor.shape[-1]),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": index,
                "instance": str(index),
                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                "img_mask": torch.tensor([True]),
                "ray_mask": torch.tensor([False]),
                "update": torch.tensor([update]),
                "reset": torch.tensor([index == 0]),
            }
        )
    return views


def _build_sequence(events: list[dict], sequence: str) -> tuple[list[str], list[bool], dict[str, list[int]]]:
    paths, updates, queries = [], [], {}
    selected = sorted(
        (event for event in events if event["sequence"] == sequence),
        key=lambda event: (event["timestamp"], event["event_id"]),
    )
    for event in selected:
        for frame in event["context"]:
            paths.append(frame["rgb"])
            updates.append(True)
        if event["is_revisit_target"]:
            positions = []
            for frame in event["query"]:
                positions.append(len(paths))
                paths.append(frame["rgb"])
                updates.append(False)
            queries[event["event_id"]] = positions
    return paths, updates, queries


def _summary(rows: list[dict], mode: str) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    return {
        "targets": len(rows),
        "sequences": len(sequences),
        **{
            metric: float(
                np.mean(
                    [
                        np.mean([row[mode][metric] for row in rows if row["sequence"] == sequence])
                        for sequence in sequences
                    ]
                )
            )
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-036_cut3r_ttt3r_baselines_v10.yaml")
    parser.add_argument("--confirm-baseline-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_baseline_evaluation:
        raise SystemExit("EXP-036 requires explicit baseline confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-036 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-036 result already exists")
    manifest_path = Path(config["data"]["manifest"])
    checkpoint = Path(config["baseline"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(checkpoint) == config["baseline"]["checkpoint_sha256"]
    ):
        raise RuntimeError("EXP-036 frozen input hash mismatch")

    repository = Path(config["baseline"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.image import load_images_for_eval

    events = json.loads(manifest_path.read_text())
    event_by_id = {event["event_id"]: event for event in events}
    sequences = sorted({event["sequence"] for event in events})
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to("cuda")
    model.eval().requires_grad_(False)
    rows_by_target: dict[str, dict] = {}
    runtime = {}
    side = int(config["depth"]["grid_side"])

    with torch.no_grad():
        for mode in config["baseline"]["modes"]:
            model.config.model_update_type = mode
            mode_start = time.perf_counter()
            mode_frames = 0
            for sequence in sequences:
                paths, updates, query_positions = _build_sequence(events, sequence)
                images = load_images_for_eval(
                    paths,
                    size=int(config["baseline"]["image_size"]),
                    verbose=False,
                    crop=bool(config["baseline"]["crop"]),
                )
                views = _views(images, updates)
                torch.cuda.synchronize()
                sequence_start = time.perf_counter()
                predictions = model.forward_recurrent_lighter(views, device="cuda", ret_state=False)
                torch.cuda.synchronize()
                sequence_seconds = time.perf_counter() - sequence_start
                mode_frames += len(views)

                for event_id, positions in query_positions.items():
                    event = event_by_id[event_id]
                    depth_predictions = []
                    output_height = output_width = None
                    for position in positions:
                        points = predictions[position]["pts3d_in_self_view"]
                        depth = points[0, ..., 2].float()
                        output_height, output_width = depth.shape
                        pooled = F.interpolate(
                            depth[None, None], size=(side, side), mode="area"
                        )[0, 0]
                        depth_predictions.append(pooled)
                    prediction = torch.stack(depth_predictions).cpu().numpy()
                    target, valid = _query_depth_gt(event, side, config)
                    original_width, original_height = Image.open(event["query"][0]["rgb"]).size
                    fx, fy, cx, cy = event["intrinsics_fx_fy_cx_cy"]
                    intrinsics = np.tile(
                        np.asarray(
                            [
                                fx * output_width / original_width,
                                fy * output_height / original_height,
                                cx * output_width / original_width,
                                cy * output_height / original_height,
                            ],
                            dtype=np.float64,
                        ),
                        (len(positions), 1),
                    )
                    metrics = _depth_metrics(
                        prediction,
                        target,
                        valid,
                        intrinsics,
                        image_size=(output_height, output_width),
                        minimum_cells=int(config["depth"]["minimum_cells_per_view"]),
                    )
                    if metrics is None:
                        raise RuntimeError(f"no valid depth metrics for {mode}:{event_id}")
                    row = rows_by_target.setdefault(
                        event_id, {"target": event_id, "sequence": sequence}
                    )
                    row[mode] = metrics
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "sequence": sequence,
                            "frames": len(views),
                            "targets": len(query_positions),
                            "seconds": sequence_seconds,
                        }
                    ),
                    flush=True,
                )
                del predictions, views, images
                gc.collect()
                torch.cuda.empty_cache()
            runtime[mode] = {
                "frames": mode_frames,
                "seconds_including_preprocessing": time.perf_counter() - mode_start,
            }

    rows = [rows_by_target[key] for key in sorted(rows_by_target)]
    modes = list(config["baseline"]["modes"])
    summaries = {mode: _summary(rows, mode) for mode in modes}
    ours = json.loads(Path(config["data"]["ours_result"]).read_text())
    summaries["revisit3d_current"] = ours["summaries"]["current"]
    summaries["revisit3d_full"] = ours["summaries"]["full"]
    checks = {}
    for mode in modes:
        checks[f"{mode}_coverage"] = (
            summaries[mode]["targets"] >= int(config["success"]["minimum_targets_per_mode"])
            and summaries[mode]["sequences"] >= int(config["success"]["minimum_sequences_per_mode"])
        )
        checks[f"{mode}_metrics_finite"] = all(
            np.isfinite(summaries[mode][metric]) for metric in METRICS
        )
    result = {
        "experiment": "EXP-036",
        "stage": "matched_cut3r_ttt3r_tum_baselines",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "query_update": False,
        "sequence_reset_only": True,
        "summaries": summaries,
        "runtime": runtime,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "rows": rows,
        "comparison_limitations": [
            "architectures_and_training_data_differ",
            "cut3r_ttt3r_use_official_512_preprocessing_while_revisit3d_uses_224",
            "only_three_imbalanced_tum_sequences",
        ],
        "final_model_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"summaries": summaries, "runtime": runtime, "gate": result["registered_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
