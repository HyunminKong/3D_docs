#!/usr/bin/env python3
"""No-fit official FastVGGT geometry-carrier diagnostic on exposed TUM."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenVGGTDepthTeacher
from revisit3d.scripts.cache_exp035_tum_geometry import _frames, _load_views
from revisit3d.scripts.evaluate_exp010_absolute_geometry import _depth_metrics
from revisit3d.scripts.evaluate_exp035_tum_zero_shot import METRICS, PRIMARY, _query_depth_gt
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _summary(rows: list[dict]) -> dict:
    sequences = sorted({row["sequence"] for row in rows})
    return {
        "targets": len(rows),
        "sequences": len(sequences),
        **{
            metric: float(
                np.mean(
                    [
                        np.mean([row["official_fastvggt"][metric] for row in rows if row["sequence"] == sequence])
                        for sequence in sequences
                    ]
                )
            )
            for metric in METRICS
        },
    }


def _per_sequence(rows: list[dict]) -> dict:
    return {
        sequence: {
            "targets": len(selected),
            **{
                metric: float(np.mean([row["official_fastvggt"][metric] for row in selected]))
                for metric in METRICS
            },
        }
        for sequence in sorted({row["sequence"] for row in rows})
        for selected in [[row for row in rows if row["sequence"] == sequence]]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-037_official_fastvggt_carrier_v10.yaml")
    parser.add_argument("--confirm-exposed-carrier-diagnostic", action="store_true")
    args = parser.parse_args()
    if not args.confirm_exposed_carrier_diagnostic:
        raise SystemExit("EXP-037 requires explicit exposed-diagnostic confirmation")
    if not torch.cuda.is_available():
        raise SystemExit("EXP-037 requires CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-037 result already exists")

    manifest_path = Path(config["data"]["manifest"])
    custom_path = Path(config["data"]["custom_result"])
    external_path = Path(config["data"]["external_result"])
    checkpoint_path = Path(config["foundation"]["checkpoint"])
    hashes_match = (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(custom_path) == config["data"]["custom_result_sha256"]
        and _sha256(external_path) == config["data"]["external_result_sha256"]
        and _sha256(checkpoint_path) == config["foundation"]["checkpoint_sha256"]
    )
    if not hashes_match:
        raise RuntimeError("EXP-037 frozen input hash mismatch")
    if config["foundation"]["input_policy"] != "query_views_only":
        raise RuntimeError("EXP-037 permits only the registered query-only carrier interface")
    if config["foundation"]["query_updates_state"] is not False:
        raise RuntimeError("EXP-037 query frames may not update streaming state")

    events = json.loads(manifest_path.read_text())
    targets = [event for event in events if event["is_revisit_target"]]
    height = int(config["foundation"]["image_height"])
    width = int(config["foundation"]["image_width"])
    side = int(config["depth"]["grid_side"])
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FrozenVGGTDepthTeacher(
        checkpoint_path, repo_root=config["foundation"]["repository"]
    ).to(device)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    with torch.no_grad():
        for index, event in enumerate(targets):
            raw = _load_views(_frames(event, "query"), height=height, width=width)
            rgb = raw["rgb"].to(device)
            prediction = model(rgb, output_size=(side, side))["depth"][0]

            # Dense TUM depth remains evaluation-only and is opened after the
            # prediction. Dataset intrinsics match the square resize used by
            # every Revisit3D TUM evaluation.
            target, valid = _query_depth_gt(event, side, config)
            metrics = _depth_metrics(
                prediction.float().cpu().numpy(),
                target,
                valid,
                raw["intrinsics"][0].numpy(),
                image_size=(height, width),
                minimum_cells=int(config["depth"]["minimum_cells_per_view"]),
            )
            if metrics is None:
                raise RuntimeError(f"no valid depth metrics for {event['event_id']}")
            rows.append(
                {
                    "target": event["event_id"],
                    "sequence": event["sequence"],
                    "official_fastvggt": metrics,
                }
            )
            if index % 10 == 0 or index + 1 == len(targets):
                print(json.dumps({"evaluated": index + 1, "total": len(targets)}), flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    summary = _summary(rows)
    per_sequence = _per_sequence(rows)
    custom = json.loads(custom_path.read_text())["summaries"]["full"]
    external = json.loads(external_path.read_text())["summaries"]
    reference_name = str(config["decision"]["reference_mode"])
    reference = external[reference_name]
    ratios = {
        metric: float(summary[metric] / reference[metric]) for metric in PRIMARY
    }
    improvements_over_custom = {
        metric: float((custom[metric] - summary[metric]) / custom[metric]) for metric in PRIMARY
    }
    checks = {
        "exact_coverage": summary["targets"] == int(config["success"]["exact_targets"])
        and summary["sequences"] == int(config["success"]["exact_sequences"]),
        "all_metrics_finite": all(np.isfinite(summary[metric]) for metric in METRICS),
        "better_than_custom_all_primary": all(summary[metric] < custom[metric] for metric in PRIMARY),
        "within_registered_reference_ratio_all_primary": all(
            ratios[metric] <= float(config["decision"]["maximum_error_ratio_to_reference"])
            for metric in PRIMARY
        ),
    }
    viable = all(checks.values())
    result = {
        "experiment": "EXP-037",
        "stage": "official_fastvggt_exposed_carrier_diagnostic",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "purpose": config["purpose"],
        "tum_fit_performed": False,
        "query_input_used_for_prediction": True,
        "query_updates_state": False,
        "input_policy": config["foundation"]["input_policy"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "summary": summary,
        "per_sequence": per_sequence,
        "references": {
            "revisit3d_custom_full": custom,
            reference_name: reference,
        },
        "error_ratios_to_reference": ratios,
        "relative_improvements_over_custom": improvements_over_custom,
        "runtime": {
            "targets": len(rows),
            "seconds": elapsed,
            "seconds_per_target": elapsed / max(len(rows), 1),
            "peak_gpu_allocated_bytes": peak_gpu_bytes,
        },
        "registered_gate": {"checks": checks, "passed": viable},
        "registered_decision": (
            config["decision"]["if_pass"] if viable else config["decision"]["if_fail"]
        ),
        "final_test_evidence": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "summary": summary,
                "ratios": ratios,
                "improvements_over_custom": improvements_over_custom,
                "gate": result["registered_gate"],
                "decision": result["registered_decision"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
