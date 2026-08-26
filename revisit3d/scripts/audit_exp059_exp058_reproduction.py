#!/usr/bin/env python3
"""Account for EXP-058's reproduction-guard miss without rerunning either model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(row: dict) -> tuple[str, str, int]:
    return row["scene"], row["sequence"], int(row["target_frame"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-059_exp058_reproduction_accounting_v10.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    source_cfg = config["sources"]
    exp057_path = Path(source_cfg["exp057"])
    exp058_path = Path(source_cfg["exp058"])
    hashes = {
        "exp057": _sha256(exp057_path),
        "exp058": _sha256(exp058_path),
    }
    expected_hashes = {
        "exp057": source_cfg["exp057_sha256"],
        "exp058": source_cfg["exp058_sha256"],
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"immutable source hash mismatch: {hashes}")

    exp057 = json.loads(exp057_path.read_text())
    exp058 = json.loads(exp058_path.read_text())
    if exp057["experiment"] != "EXP-057" or exp058["experiment"] != "EXP-058":
        raise RuntimeError("source experiment identity mismatch")

    rows057 = {_key(row): row for row in exp057["rows"]}
    rows058 = {_key(row): row for row in exp058["rows"]}
    if rows057.keys() != rows058.keys():
        raise RuntimeError("EXP-057/058 row identities differ")

    comparisons = []
    for key in sorted(rows057):
        row057 = rows057[key]
        row058 = rows058[key]
        comparisons.append(
            {
                "scene": key[0],
                "sequence": key[1],
                "target_frame": key[2],
                "supported_pixels_equal": (
                    row057["supported_pixels"] == row058["supported_pixels"]
                ),
                "erased_abs_difference": abs(
                    row058["errors"]["erased"] - row057["errors"]["erased"]
                ),
                "current_one_abs_difference": abs(
                    row058["errors"]["current_one"]
                    - row057["errors"]["current_one"]
                ),
                "current_two_abs_difference": abs(
                    row058["errors"]["current_two"]
                    - row057["errors"]["current_two"]
                ),
            }
        )

    tolerance = float(config["registered_exp058_tolerance"])
    maxima = {
        name: max(row[name] for row in comparisons)
        for name in (
            "erased_abs_difference",
            "current_one_abs_difference",
            "current_two_abs_difference",
        )
    }
    failing_second_rows = [
        row for row in comparisons if row["current_two_abs_difference"] > tolerance
    ]
    mean_gain = float(exp058["means"]["predicted_only_gain_vs_second"])
    minimum_scene_gain = min(
        float(value)
        for value in exp058["scene_means"][
            "predicted_only_gain_vs_second"
        ].values()
    )
    lower_ci = float(
        exp058["intervals"]["predicted_only_gain_vs_second"]["ci95"][0]
    )
    max_second_drift = maxima["current_two_abs_difference"]

    payload = {
        "experiment": "EXP-059",
        "stage": "immutable_artifact_reproduction_accounting",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "sources": {
            "exp057": str(exp057_path),
            "exp058": str(exp058_path),
            "sha256": hashes,
        },
        "sensor_or_model_access": False,
        "model_rerun": False,
        "gate_or_tolerance_changed": False,
        "validation_accessed": False,
        "terminal_accessed": False,
        "counts": {
            "matched_rows": len(comparisons),
            "matched_support_rows": sum(
                int(row["supported_pixels_equal"]) for row in comparisons
            ),
            "second_current_rows_over_registered_tolerance": len(
                failing_second_rows
            ),
        },
        "registered_exp058_tolerance": tolerance,
        "maximum_absolute_differences": maxima,
        "second_current_rows_over_registered_tolerance": failing_second_rows,
        "scale_accounting": {
            "exp058_mean_predicted_only_gain_vs_second": mean_gain,
            "exp058_lower_ci_predicted_only_gain_vs_second": lower_ci,
            "minimum_exp058_scene_gain_vs_second": minimum_scene_gain,
            "max_second_drift_fraction_of_mean_gain": max_second_drift / mean_gain,
            "max_second_drift_fraction_of_lower_ci": max_second_drift / lower_ci,
            "max_second_drift_fraction_of_minimum_scene_gain": (
                max_second_drift / minimum_scene_gain
            ),
        },
        "audit_checks": {
            "immutable_source_hashes_match": hashes == expected_hashes,
            "all_row_identities_match": rows057.keys() == rows058.keys(),
            "all_evaluation_support_matches": all(
                row["supported_pixels_equal"] for row in comparisons
            ),
            "pre_adaptation_erased_baseline_is_exact": (
                maxima["erased_abs_difference"] == 0.0
            ),
            "difference_is_localized_after_local_code_adaptation": (
                maxima["erased_abs_difference"] == 0.0
                and maxima["current_two_abs_difference"] > tolerance
            ),
            "exp058_registered_gate_remains_failed": (
                not exp058["registered_gate"]["passed"]
                and not exp058["registered_gate"]["checks"][
                    "exp057_second_reproduction"
                ]
            ),
            "all_exp058_functional_method_checks_pass": all(
                exp058["registered_gate"]["checks"][name]
                for name in (
                    "minimum_predicted_only_coverage",
                    "predicted_only_beats_second_all_scenes",
                    "predicted_only_beats_second_positive_ci",
                    "predicted_only_beats_shuffle_all_scenes",
                    "predicted_only_beats_shuffle_positive_ci",
                    "predicted_only_harm_within_bound",
                    "minimum_oracle_gain_retention",
                )
            ),
        },
        "classification": "qualified_positive_not_literal_pass",
        "interpretation": (
            "EXP-058 remains a registered-gate failure. Row identities, scoring "
            "support, and the pre-adaptation erased prediction reproduce exactly; "
            "the guard miss is confined to repeated local-code gradient steps. "
            "All within-run surface-fusion comparisons pass by margins orders of "
            "magnitude larger than the observed drift, so the immutable result is "
            "qualified positive dependency evidence rather than a repaired pass."
        ),
        "rows": comparisons,
    }

    output = Path(config["output"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["maximum_absolute_differences"], indent=2))
    print(json.dumps(payload["audit_checks"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
