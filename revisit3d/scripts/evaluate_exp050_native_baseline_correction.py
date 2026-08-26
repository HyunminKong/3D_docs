#!/usr/bin/env python3
"""Re-run matched native CUT3R/TTT3R after EXP-050's stale-baseline guard."""
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
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import (
    METRICS,
    _build_sequence,
    _summary,
    _views,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-050_native_baseline_correction_v11.yaml"
    )
    parser.add_argument("--confirm-native-baseline-correction", action="store_true")
    args = parser.parse_args()
    if not args.confirm_native_baseline_correction or not torch.cuda.is_available():
        raise SystemExit("EXP-050 native correction requires confirmation and CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-050 native correction already exists")
    manifest_path = Path(config["data"]["manifest"])
    inventory_path = Path(config["data"]["input_inventory"])
    checkpoint = Path(config["baseline"]["checkpoint"])
    inventory = json.loads(inventory_path.read_text())
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(inventory_path) == config["data"]["input_inventory_file_sha256"]
        and inventory["inventory_sha256"] == config["data"]["input_inventory_sha256"]
        and _sha256(checkpoint) == config["baseline"]["checkpoint_sha256"]
        and config["baseline"]["modes"] == ["cut3r", "ttt3r"]
        and config["baseline"]["query_update"] is False
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-050 native correction contract failed")

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
    processed_frames = {}

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
                predictions, _ = model.forward_recurrent_lighter(
                    views, device="cuda", ret_state=False
                )
                mode_frames += len(views)
                for event_id, positions in query_positions.items():
                    event = event_by_id[event_id]
                    depth_predictions = []
                    output_height = output_width = None
                    for position in positions:
                        depth = predictions[position]["pts3d_in_self_view"][0, ..., 2].float()
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
                        raise RuntimeError(f"EXP-050 no metrics for {mode}:{event_id}")
                    rows_by_target.setdefault(
                        event_id, {"target": event_id, "sequence": sequence}
                    )[mode] = metrics
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "sequence": sequence,
                            "frames": len(views),
                            "targets": len(query_positions),
                        }
                    ),
                    flush=True,
                )
                del predictions, views, images
                gc.collect()
                torch.cuda.empty_cache()
            processed_frames[mode] = mode_frames
            runtime[mode] = time.perf_counter() - mode_start

    rows = [rows_by_target[key] for key in sorted(rows_by_target)]
    summaries = {
        mode: _summary(rows, mode) for mode in config["baseline"]["modes"]
    }
    checks = {
        "exact_coverage": len(rows) == int(config["data"]["exact_targets"])
        and len(sequences) == int(config["data"]["exact_sequences"])
        and all(
            count == int(config["data"]["exact_stream_frames_per_mode"])
            for count in processed_frames.values()
        ),
        "finite": all(
            np.isfinite(summary[metric])
            for summary in summaries.values()
            for metric in METRICS
        ),
    }
    result = {
        "experiment": "EXP-050",
        "stage": config["purpose"],
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "input_inventory_sha256": inventory["inventory_sha256"],
        "summaries": summaries,
        "runtime_seconds_including_preprocessing": runtime,
        "processed_frames": processed_frames,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "query_update": False,
        "sequence_reset_only": True,
        "fitting_performed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"summaries": summaries, "gate": result["registered_gate"]}, indent=2))


if __name__ == "__main__":
    main()
