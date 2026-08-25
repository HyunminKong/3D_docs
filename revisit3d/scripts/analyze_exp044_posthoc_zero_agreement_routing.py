#!/usr/bin/env python3
"""Record the post-hoc EXP-044 zero-agreement routing diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _scene_balanced(rows: list[dict], key: str) -> float:
    scenes = sorted({row["scene"] for row in rows})
    return float(
        np.mean(
            [np.mean([row[key] for row in rows if row["scene"] == scene]) for scene in scenes]
        )
    )


def _bootstrap_gain(rows: list[dict], *, draws: int, seed: int) -> dict:
    scenes = sorted({row["scene"] for row in rows})
    values = np.asarray(
        [
            np.mean(
                [
                    row["target_current_loss"] - row["selected_loss"]
                    for row in rows
                    if row["scene"] == scene
                ]
            )
            for scene in scenes
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "positive_scenes": int((values > 0).sum()),
        "scenes": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-044_posthoc_zero_agreement_routing_v10.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    input_path = Path(config["input"]["result"])
    output = Path(config["output"]["result"])
    if output.exists():
        raise RuntimeError("EXP-044 result already exists")
    if _sha256(input_path) != config["input"]["result_sha256"]:
        raise RuntimeError("EXP-044 input hash mismatch")
    source = json.loads(input_path.read_text())
    if source["validation_accessed"] or source["terminal_accessed"]:
        raise RuntimeError("EXP-044 requires train-internal rows only")
    base_rows = source[config["input"]["audit_branch"]]["rows"]
    threshold = float(config["policy"]["threshold"])
    policies = {}
    for name in ("ungated", "zero_agreement", "oracle_fallback"):
        rows = []
        for row in base_rows:
            if name == "ungated":
                accepted = True
            elif name == "zero_agreement":
                accepted = row["source_target_code_agreement"] > threshold
            else:
                accepted = row["target_full_loss"] < row["target_current_loss"]
            selected_loss = row["target_full_loss"] if accepted else row["target_current_loss"]
            rows.append(
                {
                    "pair_id": row["pair_id"],
                    "scene": row["scene"],
                    "accepted": accepted,
                    "agreement": row["source_target_code_agreement"],
                    "target_current_loss": row["target_current_loss"],
                    "target_full_loss": row["target_full_loss"],
                    "selected_loss": selected_loss,
                }
            )
        policies[name] = {
            "selected_loss": _scene_balanced(rows, "selected_loss"),
            "gain_over_current": _scene_balanced(rows, "target_current_loss")
            - _scene_balanced(rows, "selected_loss"),
            "acceptance_fraction": float(np.mean([row["accepted"] for row in rows])),
            "harm_fraction": float(
                np.mean(
                    [row["selected_loss"] > row["target_current_loss"] for row in rows]
                )
            ),
            "uncertainty": _bootstrap_gain(
                rows,
                draws=int(config["analysis"]["bootstrap_draws"]),
                seed=int(config["analysis"]["bootstrap_seed"]),
            ),
            "rows": rows,
        }
    agreement = np.asarray(
        [row["source_target_code_agreement"] for row in base_rows], dtype=np.float64
    )
    utility = np.asarray(
        [row["target_current_loss"] - row["target_full_loss"] for row in base_rows],
        dtype=np.float64,
    )
    result = {
        "experiment": "EXP-044",
        "stage": "posthoc_zero_agreement_routing_diagnostic",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "posthoc": True,
        "threshold_fitted": False,
        "threshold": threshold,
        "pairs": len(base_rows),
        "scenes": len({row["scene"] for row in base_rows}),
        "agreement_utility_pearson": float(np.corrcoef(agreement, utility)[0, 1]),
        "policies": policies,
        "validation_accessed": False,
        "terminal_accessed": False,
        "confirmatory_claim_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "agreement_utility_pearson": result["agreement_utility_pearson"],
                "policies": {
                    name: {key: value[key] for key in ("gain_over_current", "acceptance_fraction", "harm_fraction", "uncertainty")}
                    for name, value in policies.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
