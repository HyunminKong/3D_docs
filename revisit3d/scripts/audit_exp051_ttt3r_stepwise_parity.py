#!/usr/bin/env python3
"""Train-RGB-only native/step-wise TTT3R parity audit for EXP-051."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import _views
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


KEYS = ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")


def _maximum_error(left: dict, right: dict) -> tuple[float, dict[str, float]]:
    errors = {
        key: float(
            (
                left[key].detach().cpu().float()
                - right[key].detach().cpu().float()
            )
            .abs()
            .max()
        )
        for key in KEYS
        if key in left and key in right
    }
    return max(errors.values(), default=0.0), errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/EXP-051_ttt3r_metric_aligned_prerequisites_v10.yaml",
    )
    parser.add_argument("--confirm-train-rgb-parity", action="store_true")
    args = parser.parse_args()
    if not args.confirm_train_rgb_parity or not torch.cuda.is_available():
        raise SystemExit("EXP-051 parity requires confirmation and CUDA")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    carrier_config = config["carrier"]
    output = Path(config["output"]["parity_result"])
    inventory_path = Path(config["output"]["inventory"])
    train_manifest_path = Path(config["output"]["train_manifest"])
    checkpoint = Path(carrier_config["checkpoint"])
    if output.exists():
        raise RuntimeError("EXP-051 parity artifact already exists")
    inventory = json.loads(inventory_path.read_text())
    train_manifest = json.loads(train_manifest_path.read_text())
    if not (
        inventory["registered_gate"]["passed"]
        and inventory["terminal_accessed"] is False
        and train_manifest["role"] == "train"
        and carrier_config["mode"] == "ttt3r"
        and _sha256(checkpoint) == carrier_config["checkpoint_sha256"]
    ):
        raise RuntimeError("EXP-051 frozen input contract failed")

    sequence = carrier_config["parity_sequence"]
    if sequence.split("/", 1)[0] not in train_manifest["scenes"]:
        raise RuntimeError("EXP-051 parity sequence is not in the train role")
    count = int(carrier_config["parity_train_frames"])
    root = Path(config["data"]["root"])
    paths = [
        str(root / sequence / f"frame-{index:06d}.color.png")
        for index in range(count)
    ]
    if not all(Path(path).is_file() for path in paths):
        raise RuntimeError("EXP-051 parity RGB path is missing")

    repository = Path(carrier_config["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    images = load_images_for_eval(
        paths,
        size=int(carrier_config["image_size"]),
        verbose=False,
        crop=bool(carrier_config["crop"]),
    )
    views = _views(images, [True] * count)
    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
        update_type="ttt3r",
    ).cuda()
    carrier.eval().requires_grad_(False)

    with torch.no_grad():
        native, _ = carrier.model.forward_recurrent_lighter(
            views, device="cuda", ret_state=False
        )
        state = None
        rows = []
        for index, view in enumerate(views):
            stepwise, state, _ = carrier.step(view, state, code=None)
            maximum, errors = _maximum_error(native[index], stepwise)
            rows.append(
                {
                    "frame": index,
                    "maximum_abs_difference": maximum,
                    "by_output": errors,
                }
            )

    maximum = max(row["maximum_abs_difference"] for row in rows)
    threshold = float(config["success"]["maximum_native_stepwise_abs_difference"])
    checks = {
        "exact_frame_coverage": len(rows) == count,
        "finite": all(
            torch.isfinite(torch.tensor(row["maximum_abs_difference"]))
            for row in rows
        ),
        "native_stepwise_parity": maximum <= threshold,
        "train_rgb_only": True,
        "no_ground_truth_access": True,
        "no_terminal_access": True,
    }
    result = {
        "experiment": "EXP-051",
        "stage": "ttt3r_stepwise_zero_code_parity",
        "config": str(config_path),
        "inventory_sha256": _sha256(inventory_path),
        "train_manifest_sha256": _sha256(train_manifest_path),
        "sequence": sequence,
        "frames": count,
        "maximum_abs_difference": maximum,
        "threshold": threshold,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "basis_fit_performed": False,
        "ground_truth_accessed": False,
        "terminal_accessed": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    del carrier, state, native, views, images
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
