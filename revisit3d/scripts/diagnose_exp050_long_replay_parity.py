#!/usr/bin/env python3
"""GT-free long-stream parity diagnosis for the failed EXP-050 replay guard."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenCUT3RCarrier
from revisit3d.scripts.evaluate_exp036_cut3r_ttt3r_baselines import (
    _build_sequence,
    _views,
)
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


KEYS = ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose")


def _error(left: dict, right: dict) -> float:
    return max(
        float((left[key].detach().cpu() - right[key].detach().cpu()).abs().max())
        for key in KEYS
        if key in left and key in right
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-050_current_only_exact_meta_tum_v10.yaml"
    )
    parser.add_argument("--confirm-gt-free-parity-diagnosis", action="store_true")
    args = parser.parse_args()
    if not args.confirm_gt_free_parity_diagnosis or not torch.cuda.is_available():
        raise SystemExit("EXP-050 long parity diagnosis requires confirmation and CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    manifest_path = Path(config["data"]["manifest"])
    checkpoint = Path(config["carrier"]["checkpoint"])
    if not (
        _sha256(manifest_path) == config["data"]["manifest_sha256"]
        and _sha256(checkpoint) == config["carrier"]["checkpoint_sha256"]
        and config["data"]["terminal_access"] is False
    ):
        raise RuntimeError("EXP-050 parity diagnosis input contract failed")
    repository = Path(config["carrier"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    from dust3r.utils.image import load_images_for_eval

    events = json.loads(manifest_path.read_text())
    sequences = sorted({event["sequence"] for event in events})
    carrier = FrozenCUT3RCarrier(
        checkpoint,
        repository=repository,
        code_dim=int(config["plasticity"]["code_dim"]),
    ).cuda()
    carrier.eval().requires_grad_(False)
    rows = []
    with torch.no_grad():
        for sequence in sequences:
            paths, updates, query_positions = _build_sequence(events, sequence)
            query_set = {
                position for positions in query_positions.values() for position in positions
            }
            images = load_images_for_eval(
                paths,
                size=int(config["carrier"]["image_size"]),
                verbose=False,
                crop=bool(config["carrier"]["crop"]),
            )
            views = _views(images, updates)
            native, _ = carrier.model.forward_recurrent_lighter(
                views, device="cuda", ret_state=False
            )
            state = None
            errors = []
            for index, view in enumerate(views):
                custom, state, _ = carrier.step(view, state)
                errors.append(_error(native[index], custom))
            nonzero = [index for index, value in enumerate(errors) if value != 0.0]
            query_errors = [errors[index] for index in sorted(query_set)]
            row = {
                "sequence": sequence,
                "frames": len(views),
                "queries": len(query_set),
                "maximum_error": max(errors),
                "first_nonzero_frame": nonzero[0] if nonzero else None,
                "nonzero_frames": len(nonzero),
                "maximum_query_error": max(query_errors) if query_errors else 0.0,
                "mean_query_error": float(np.mean(query_errors)) if query_errors else 0.0,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del images, views, native, state
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "experiment": "EXP-050",
        "stage": "gt_free_long_replay_parity_diagnosis",
        "config": args.config,
        "rows": rows,
        "ground_truth_accessed": False,
        "model_or_parameter_selection_performed": False,
        "terminal_accessed": False,
    }
    output = Path("revisit3d/results/EXP-050/long_replay_parity_diagnosis_v10.json")
    if output.exists():
        raise RuntimeError("EXP-050 parity diagnosis already exists")
    output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
