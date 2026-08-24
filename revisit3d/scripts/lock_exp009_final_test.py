#!/usr/bin/env python3
"""Freeze the validation-selected capacity into the final EXP-009 test artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-009_final_test_v24.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["artifact"])
    result_path = Path(config["output"]["lock_result"])
    if output.exists() or result_path.exists():
        raise RuntimeError("final test lock output already exists")
    selection = json.loads(Path(config["source"]["capacity_selection_result"]).read_text())
    train_lock = json.loads(Path(config["source"]["train_lock_result"]).read_text())
    train_artifact_path = Path(config["source"]["train_artifact"])
    if not (
        selection.get("split") == "val"
        and selection.get("validation_accessed") is True
        and selection.get("test_accessed") is False
        and selection.get("selected_capacity") == int(config["bank"]["capacity"]) == 64
        and selection.get("registered_gate", {}).get("passed") is True
        and train_lock.get("artifact_sha256") == _sha256(train_artifact_path)
        and train_lock.get("test_accessed") is False
    ):
        raise RuntimeError("final capacity/model lock contract failed")
    artifact = joblib.load(train_artifact_path)
    artifact["protocol_revision"] = config["protocol_revision"]
    artifact["bank_capacity"] = int(config["bank"]["capacity"])
    artifact["bank_retention"] = config["bank"]["retention"]
    artifact["candidate_count"] = int(config["bank"]["candidate_count"])
    artifact["validation_selected_capacity"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    result = {
        "experiment": "EXP-009", "stage": "stage15_final_test_lock",
        "protocol_revision": config["protocol_revision"], "split": "train_val_lock",
        "validation_accessed": True, "test_accessed": False,
        "query_or_future_router_input": False, "config": str(config_path),
        "artifact": str(output), "artifact_sha256": _sha256(output),
        "source_artifact_sha256": train_lock["artifact_sha256"],
        "selected_capacity": 64, "retention": config["bank"]["retention"],
        "candidate_count": int(config["bank"]["candidate_count"]),
        "capacity_selection_result": config["source"]["capacity_selection_result"],
        "test_accessed_after_lock": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
