#!/usr/bin/env python3
"""Recompute only registered EXP-050 gates with the matched native baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import METRICS
from revisit3d.scripts.evaluate_exp050_current_only_exact_meta_tum import (
    PRIMARY,
    _bootstrap,
    _summary,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-050_current_only_exact_meta_tum_v11.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-050 corrected result already exists")
    registered_config_path = Path(config["source"]["registered_config"])
    method_path = Path(config["source"]["method_result"])
    native_path = Path(config["source"]["native_baseline_result"])
    parity_path = Path(config["source"]["long_parity_result"])
    if not (
        _sha256(registered_config_path)
        == config["source"]["registered_config_sha256"]
        and _sha256(method_path) == config["source"]["method_result_sha256"]
        and _sha256(native_path)
        == config["source"]["native_baseline_result_sha256"]
        and _sha256(parity_path) == config["source"]["long_parity_result_sha256"]
    ):
        raise RuntimeError("EXP-050 correction source hash mismatch")
    method = json.loads(method_path.read_text())
    native = json.loads(native_path.read_text())
    parity = json.loads(parity_path.read_text())
    registered = yaml.safe_load(registered_config_path.read_text())
    if not (
        method["targets"] == 111
        and method["processed_frames"] == 2228
        and not method["terminal_accessed"]
        and native["registered_gate"]["passed"]
        and native["input_inventory_sha256"]
        == config["source"]["input_inventory_sha256"]
        and all(row["maximum_error"] == 0.0 for row in parity["rows"])
        and not parity["ground_truth_accessed"]
        and config["analysis"] == registered["analysis"]
        and config["success"] == registered["success"]
    ):
        raise RuntimeError("EXP-050 correction scope changed")

    native_by_id = {row["target"]: row for row in native["rows"]}
    rows = []
    for source in method["rows"]:
        matched = native_by_id[source["target"]]
        rows.append(
            {
                **{key: value for key, value in source.items() if key != "exp036_cut3r"},
                "native_cut3r": matched["cut3r"],
                "ttt3r": matched["ttt3r"],
            }
        )
    reproduction_max = max(
        abs(row["cut3r"][metric] - row["native_cut3r"][metric])
        for row in rows
        for metric in METRICS
    )
    summaries = {
        policy: _summary(rows, policy)
        for policy in (
            "cut3r",
            "generic_one",
            "exact_one",
            "exact_two",
            "native_cut3r",
            "ttt3r",
        )
    }
    comparisons = {
        "exact_one_over_cut3r": ("cut3r", "exact_one"),
        "exact_one_over_generic": ("generic_one", "exact_one"),
        "exact_one_over_ttt3r": ("ttt3r", "exact_one"),
        "exact_two_over_exact_one": ("exact_one", "exact_two"),
        "generic_one_over_cut3r": ("cut3r", "generic_one"),
    }
    uncertainty = _bootstrap(rows, comparisons, config)
    common_checks = {
        "exact_coverage": len(rows) == 111
        and len({row["sequence"] for row in rows}) == 3,
        "long_native_step_parity": all(
            row["maximum_error"] == 0.0 for row in parity["rows"]
        ),
        "corrected_base_reproduction": reproduction_max
        <= float(config["success"]["maximum_base_reproduction_abs_difference"]),
    }
    method_checks = {
        **{
            f"exact_meta_better_cut3r_{metric}_ci95": uncertainty[
                "exact_one_over_cut3r"
            ][metric]["ci95"][0]
            > 0
            for metric in PRIMARY
        },
        **{
            f"exact_meta_better_generic_{metric}_ci95": uncertainty[
                "exact_one_over_generic"
            ][metric]["ci95"][0]
            > 0
            for metric in PRIMARY
        },
    }
    competitive_checks = {
        f"exact_meta_better_ttt3r_{metric}_ci95": uncertainty[
            "exact_one_over_ttt3r"
        ][metric]["ci95"][0]
        > 0
        for metric in PRIMARY
    }
    method_passed = all({**common_checks, **method_checks}.values())
    competitive_passed = method_passed and all(competitive_checks.values())
    result = {
        "experiment": "EXP-050",
        "stage": config["purpose"],
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "correction_scope": "native_baseline_only",
        "summaries": summaries,
        "uncertainty": uncertainty,
        "base_reproduction_max_abs": reproduction_max,
        "registered_gate": {
            "common_checks": common_checks,
            "method_checks": method_checks,
            "competitive_checks": competitive_checks,
            "method_feasibility_passed": method_passed,
            "top_tier_competitiveness_passed": competitive_passed,
        },
        "method_predictions_recomputed": False,
        "fitting_performed": False,
        "memory_active": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "summaries": summaries,
                "uncertainty": uncertainty,
                "base_reproduction_max_abs": reproduction_max,
                "gate": result["registered_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
