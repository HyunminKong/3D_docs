#!/usr/bin/env python3
"""Run the frozen EXP-070 no-fit FSM distal-interference premise."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def percentile_ci(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def decode_and_resize(blob: np.ndarray, intrinsics: np.ndarray, size: int):
    image = Image.open(io.BytesIO(blob.tobytes())).convert("RGB")
    width, height = image.size
    scale = max(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    image = image.crop((left, top, left + size, top + size))

    fx, fy, cx, cy = [float(x) for x in intrinsics]
    adjusted = np.array(
        [fx * scale, fy * scale, cx * scale - left, cy * scale - top],
        dtype=np.float32,
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    return tensor, torch.from_numpy(adjusted)


def make_batch(images: dict[int, torch.Tensor], intrinsics: torch.Tensor, frames: list[int], device):
    count = len(frames)
    return {
        "image": torch.stack([images[i] for i in frames]).unsqueeze(0).to(device),
        "fxfycxcy": intrinsics.repeat(count, 1).unsqueeze(0).to(device),
        "c2w": torch.eye(4, dtype=torch.float32).repeat(1, count, 1, 1).to(device),
        "frame_time": torch.tensor(frames, dtype=torch.long).unsqueeze(0).to(device),
    }


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    per_pixel = (pred.float() - target.float()).square().mean(dim=0)
    return float(per_pixel[mask].mean().item())


def masked_drift(pred: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> float:
    per_pixel = (pred.float() - reference.float()).square().mean(dim=0)
    return float(per_pixel[mask].mean().item())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/EXP-070_fastweight_distal_interference_v10.yaml",
    )
    parser.add_argument("--fsm-repo", type=Path, default=ROOT / "fast-spatial-mem")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Open-d4rt/data/tapvid3d/pstudio")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "revisit3d/results/EXP-070/fastweight_distal_interference_v10.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest_path = ROOT / config["source"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    observed_manifest_hash = canonical_sha256(manifest)
    expected_manifest_hash = config["source"]["manifest_canonical_sha256"]
    if observed_manifest_hash != expected_manifest_hash:
        raise RuntimeError("role manifest hash mismatch")

    names = manifest["roles"]["premise"]
    if len(names) != config["source"]["exact_sequences"]:
        raise RuntimeError("premise coverage does not match frozen count")
    closed = set(manifest["roles"]["validation"] + manifest["roles"]["terminal"])
    if set(names) & closed:
        raise RuntimeError("role overlap")

    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != config["carrier"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    repo_commit = subprocess.check_output(
        ["git", "-C", str(args.fsm_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if repo_commit != config["carrier"]["reference_commit"]:
        raise RuntimeError("FSM reference commit mismatch")

    sys.path.insert(0, str(args.fsm_repo))
    from omegaconf import OmegaConf
    from fsm.model.model_4dlvsm import FSM4DLVSM
    from utils_train import build_info, discover_fast_blocks, init_ewc_buffers

    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    model_config = OmegaConf.load(args.fsm_repo / "fsm/configs/inference/fsm_lvsm_inference.yaml")
    model_config.model.image_tokenizer.image_size = config["carrier"]["resolution"]
    model_config.model.block_config[1].params.chunk_size = config["carrier"]["fastweight_chunk_size"]
    model_config.training.actckpt = False
    model_config.training.torch_compile = False

    model = FSM4DLVSM(model_config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_message = str(model.load_state_dict(checkpoint["model"], strict=True))
    blocks = discover_fast_blocks(model)
    for _, block in blocks:
        block.anchor_mode = 2
        block.anchor_beta = float(config["carrier"]["ewc"]["anchor_beta"])

    core = [int(x) for x in config["query"]["core_A_frames"]]
    near = [int(x) for x in config["query"]["near_C_frames"]]
    distant = [int(x) for x in config["query"]["distant_B_frames"]]
    target_frame = int(config["query"]["target_frame"])
    required_frames = sorted(set(core + near + distant + [target_frame]))
    if target_frame in core or target_frame in near or target_frame in distant:
        raise RuntimeError("target leakage in frozen frame lists")

    def infer(input_batch, target_batch, elastic: bool):
        buffers = init_ewc_buffers(blocks, device=device)
        ewc = config["carrier"]["ewc"]
        info = build_info(
            blocks,
            buffers,
            batch_size=1,
            ewc_enable=elastic,
            lambda_prox=float(ewc["lambda_prox"]),
            alpha=float(ewc["fisher_alpha"]),
            mode=str(ewc["mode"]),
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(input_batch, target_batch, info=info, skip_loss=True)
        return prediction[0, 0].float().cpu()

    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for sequence_index, name in enumerate(names):
        source_path = args.data_root / f"{name}.npz"
        with np.load(source_path, allow_pickle=False) as source:
            blobs = source["images_jpeg_bytes"]
            raw_intrinsics = np.asarray(source["fx_fy_cx_cy"], dtype=np.float32)
            if len(blobs) <= max(required_frames):
                raise RuntimeError(f"{name} has insufficient frames")
            images = {}
            adjusted_intrinsics = None
            for frame in required_frames:
                image, current_intrinsics = decode_and_resize(
                    blobs[frame], raw_intrinsics, int(config["carrier"]["resolution"])
                )
                images[frame] = image
                if adjusted_intrinsics is None:
                    adjusted_intrinsics = current_intrinsics
                elif not torch.equal(adjusted_intrinsics, current_intrinsics):
                    raise RuntimeError("fixed-camera intrinsics changed during resize")

        target_batch = make_batch(images, adjusted_intrinsics, [target_frame], device)
        target = images[target_frame]
        batch_A = make_batch(images, adjusted_intrinsics, core, device)
        batch_near = make_batch(images, adjusted_intrinsics, sorted(core + near), device)
        batch_far = make_batch(images, adjusted_intrinsics, sorted(core + distant), device)

        pred_A = infer(batch_A, target_batch, elastic=True)
        pred_near = infer(batch_near, target_batch, elastic=True)
        pred_far = infer(batch_far, target_batch, elastic=True)
        pred_far_lact = infer(batch_far, target_batch, elastic=False)
        replay_A = infer(batch_A, target_batch, elastic=True)

        distant_stack = torch.stack([images[i] for i in distant])
        change_score = (distant_stack - target.unsqueeze(0)).abs().mean(dim=1).median(dim=0).values
        low = torch.quantile(change_score.flatten(), float(config["masks"]["stable_quantile"]))
        high = torch.quantile(change_score.flatten(), float(config["masks"]["changing_quantile"]))
        stable = change_score <= low
        changing = change_score >= high

        mse_A_stable = masked_mse(pred_A, target, stable)
        mse_near_stable = masked_mse(pred_near, target, stable)
        mse_far_stable = masked_mse(pred_far, target, stable)
        mse_far_lact_stable = masked_mse(pred_far_lact, target, stable)
        mse_A_changing = masked_mse(pred_A, target, changing)
        mse_far_changing = masked_mse(pred_far, target, changing)
        mse_A_all = float((pred_A - target).square().mean().item())

        row = {
            "sequence": name,
            "source_sha256": sha256_file(source_path),
            "A_only_psnr_db": float(-10.0 * np.log10(max(mse_A_all, 1e-12))),
            "exact_replay_max_abs": float((pred_A - replay_A).abs().max().item()),
            "stable_fraction": float(stable.float().mean().item()),
            "changing_fraction": float(changing.float().mean().item()),
            "mse_A_stable": mse_A_stable,
            "mse_near_stable": mse_near_stable,
            "mse_far_stable": mse_far_stable,
            "mse_far_lact_stable": mse_far_lact_stable,
            "mse_A_changing": mse_A_changing,
            "mse_far_changing": mse_far_changing,
            "stable_relative_damage_far_vs_A": (mse_far_stable - mse_A_stable) / max(mse_A_stable, 1e-12),
            "stable_relative_damage_far_vs_near": (mse_far_stable - mse_near_stable) / max(mse_near_stable, 1e-12),
            "changing_relative_damage_far_vs_A": (mse_far_changing - mse_A_changing) / max(mse_A_changing, 1e-12),
            "stable_drift_near_from_A": masked_drift(pred_near, pred_A, stable),
            "stable_drift_far_from_A": masked_drift(pred_far, pred_A, stable),
            "stable_lacet_gain_over_lact": mse_far_lact_stable - mse_far_stable,
        }
        rows.append(row)
        print(f"[{sequence_index + 1:02d}/{len(names)}] {name} "
              f"PSNR={row['A_only_psnr_db']:.3f} "
              f"far/A={100*row['stable_relative_damage_far_vs_A']:+.2f}% "
              f"far/near={100*row['stable_relative_damage_far_vs_near']:+.2f}%", flush=True)

    def vector(key):
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    far_A = vector("stable_relative_damage_far_vs_A")
    far_near = vector("stable_relative_damage_far_vs_near")
    far_drift = vector("stable_drift_far_from_A")
    near_drift = vector("stable_drift_near_from_A")
    changing_damage = vector("changing_relative_damage_far_vs_A")
    stable_damage_mean = float(far_A.mean())
    changing_damage_mean = float(changing_damage.mean())
    stable_to_changing = stable_damage_mean / max(changing_damage_mean, 1e-12)
    drift_ratio = float(far_drift.mean() / max(near_drift.mean(), 1e-12))

    stats = config["statistics"]
    reps = int(stats["bootstrap_repetitions"])
    bs_seed = int(stats["bootstrap_seed"])
    summary = {
        "sequences": len(rows),
        "mean_A_only_psnr_db": float(vector("A_only_psnr_db").mean()),
        "maximum_exact_replay_abs_difference": float(vector("exact_replay_max_abs").max()),
        "mean_stable_relative_damage_far_vs_A": stable_damage_mean,
        "stable_relative_damage_far_vs_A_ci95": percentile_ci(far_A, reps, bs_seed + 1),
        "positive_sequences_far_vs_A": int((far_A > 0).sum()),
        "mean_stable_relative_damage_far_vs_near": float(far_near.mean()),
        "stable_relative_damage_far_vs_near_ci95": percentile_ci(far_near, reps, bs_seed + 2),
        "positive_sequences_far_vs_near": int((far_near > 0).sum()),
        "far_to_near_stable_output_drift_ratio": drift_ratio,
        "stable_output_drift_difference": float((far_drift - near_drift).mean()),
        "stable_output_drift_difference_ci95": percentile_ci(far_drift - near_drift, reps, bs_seed + 3),
        "mean_changing_relative_damage_far_vs_A": changing_damage_mean,
        "stable_to_changing_damage_ratio": float(stable_to_changing),
        "mean_stable_lacet_gain_over_lact": float(vector("stable_lacet_gain_over_lact").mean()),
        "elapsed_seconds": float(time.time() - started),
        "peak_allocated_gpu_mib": float(torch.cuda.max_memory_allocated() / 2**20),
    }

    thresholds = config["gates"]
    gates = {
        "exact_premise_sequences": summary["sequences"] == int(thresholds["exact_premise_sequences"]),
        "exact_replay": summary["maximum_exact_replay_abs_difference"] <= float(thresholds["maximum_exact_replay_abs_difference"]),
        "carrier_quality": summary["mean_A_only_psnr_db"] >= float(thresholds["minimum_mean_A_only_psnr_db"]),
        "material_far_vs_A": summary["mean_stable_relative_damage_far_vs_A"] >= float(thresholds["minimum_mean_stable_relative_damage_far_vs_A"]),
        "far_vs_A_ci": summary["stable_relative_damage_far_vs_A_ci95"][0] > 0.0,
        "far_vs_A_prevalence": summary["positive_sequences_far_vs_A"] >= int(thresholds["minimum_positive_sequences_far_vs_A"]),
        "material_far_vs_near": summary["mean_stable_relative_damage_far_vs_near"] >= float(thresholds["minimum_mean_stable_relative_damage_far_vs_near"]),
        "far_vs_near_ci": summary["stable_relative_damage_far_vs_near_ci95"][0] > 0.0,
        "far_vs_near_prevalence": summary["positive_sequences_far_vs_near"] >= int(thresholds["minimum_positive_sequences_far_vs_near"]),
        "drift_ratio": summary["far_to_near_stable_output_drift_ratio"] >= float(thresholds["minimum_far_to_near_stable_output_drift_ratio"]),
        "drift_difference_ci": summary["stable_output_drift_difference_ci95"][0] > 0.0,
        "stable_damage_not_negligible": summary["stable_to_changing_damage_ratio"] >= float(thresholds["minimum_stable_to_changing_damage_ratio"]),
        "no_model_or_threshold_fit": True,
        "validation_not_accessed": True,
        "terminal_not_accessed": True,
    }

    result = {
        "experiment": "EXP-070",
        "version": "1.0",
        "status": "passed" if all(gates.values()) else "failed",
        "load_message": load_message,
        "config_path": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_canonical_sha256": observed_manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "fsm_reference_commit": repo_commit,
        "summary": summary,
        "gates": gates,
        "gate_count": {"passed": int(sum(gates.values())), "total": len(gates)},
        "sequence_rows": rows,
        "source_safety": {
            "opened_role": "premise",
            "opened_names": names,
            "validation_accessed": False,
            "terminal_accessed": False,
            "model_or_threshold_fit": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "summary": summary, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
