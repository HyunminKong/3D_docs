#!/usr/bin/env python3
"""Audit EXP-031 target accounting without rerunning or changing the terminal test."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import (
    _identifier,
    _timestamp,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-032_terminal_coverage_accounting_v10.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    terminal = json.loads(Path(config["terminal_result"]).read_text())
    if terminal["experiment"] != "EXP-031" or not terminal["terminal_accessed"]:
        raise RuntimeError("EXP-031 terminal result contract failed")

    contexts: dict[str, dict] = {}
    targets: dict[str, dict] = {}
    target_occurrences: Counter[str] = Counter()
    for index, row in enumerate(manifest):
        for tag in ("a", "b", "a_prime"):
            key = _identifier(row[tag])
            contexts.setdefault(
                key,
                {
                    "id": key,
                    "segment": row[tag],
                    "location": row["location"],
                },
            )
        key = _identifier(row["a_prime"])
        target_occurrences[key] += 1
        targets.setdefault(
            key,
            {
                "id": key,
                "segment": row["a_prime"],
                "location": row["location"],
                "component": f"component-{row['component_id']}",
                "first_manifest_index": index,
            },
        )

    metadata_cache: dict[str, dict] = {}
    scene_root = Path(config["data"]["scene_root"])
    for context in contexts.values():
        context["timestamp"] = _timestamp(
            context["segment"], scene_root, metadata_cache
        )

    causal_eligible: set[str] = set()
    empty_bank_targets: list[dict] = []
    location_counts: dict[str, dict] = {}
    for location in sorted({x["location"] for x in contexts.values()}):
        events = sorted(
            (x for x in contexts.values() if x["location"] == location),
            key=lambda x: (x["timestamp"], x["id"]),
        )
        seen = 0
        location_targets = 0
        location_eligible = 0
        for event_index, event in enumerate(events):
            if event["id"] in targets:
                location_targets += 1
                if seen:
                    causal_eligible.add(event["id"])
                    location_eligible += 1
                else:
                    target = targets[event["id"]]
                    empty_bank_targets.append(
                        {
                            "target": event["id"],
                            "location": location,
                            "scene": target["segment"]["scene"],
                            "frames": target["segment"]["frames"],
                            "event_index": event_index,
                            "reason": "first_location_event_has_empty_causal_bank",
                        }
                    )
            seen += 1
        location_counts[location] = {
            "unique_contexts": len(events),
            "unique_targets": location_targets,
            "causally_evaluable_targets": location_eligible,
        }

    evaluated = {row["target"] for row in terminal["rows"]}
    missing_eligible = sorted(causal_eligible - evaluated)
    unexpected_evaluated = sorted(evaluated - causal_eligible)
    registered_minimum = int(config["registered_minimum_targets"])
    payload = {
        "experiment": "EXP-032",
        "stage": "post_terminal_coverage_accounting",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "source_terminal_result": config["terminal_result"],
        "terminal_result_unchanged": True,
        "terminal_rerun": False,
        "counts": {
            "directional_manifest_episodes": len(manifest),
            "unique_stream_contexts": len(contexts),
            "unique_target_contexts": len(targets),
            "duplicate_directional_target_occurrences": (
                len(manifest) - len(targets)
            ),
            "empty_bank_targets": len(empty_bank_targets),
            "maximum_causally_evaluable_unique_targets": len(causal_eligible),
            "terminal_evaluated_unique_targets": len(evaluated),
            "terminal_evaluated_components": len(
                {row["component"] for row in terminal["rows"]}
            ),
            "eligible_targets_missing_from_terminal_result": len(missing_eligible),
        },
        "target_occurrence_multiplicity": {
            str(multiplicity): count
            for multiplicity, count in sorted(
                Counter(target_occurrences.values()).items()
            )
        },
        "locations": location_counts,
        "empty_bank_target_details": empty_bank_targets,
        "missing_eligible_targets": missing_eligible,
        "unexpected_evaluated_targets": unexpected_evaluated,
        "registered_coverage_gate": {
            "minimum_targets": registered_minimum,
            "maximum_possible_under_registered_evaluator": len(causal_eligible),
            "feasible": registered_minimum <= len(causal_eligible),
            "passed_as_registered": terminal["registered_gate"]["checks"][
                "coverage"
            ],
        },
        "audit_checks": {
            "all_causally_eligible_targets_evaluated": (
                evaluated == causal_eligible
            ),
            "no_metric_validity_exclusion_after_causal_eligibility": (
                not missing_eligible
            ),
            "registered_minimum_was_impossible": (
                registered_minimum > len(causal_eligible)
            ),
            "terminal_gate_remains_unmodified_and_failed": (
                not terminal["registered_gate"]["passed"]
            ),
        },
        "interpretation": (
            "EXP-031 evaluated every causally eligible unique target. The "
            "registered target minimum confused 214 directional manifest "
            "episodes with unique target contexts and exceeded the evaluator's "
            "maximum possible coverage. This audit does not repair or reclassify "
            "the registered terminal gate."
        ),
    }
    output = Path(config["output"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["audit_checks"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
