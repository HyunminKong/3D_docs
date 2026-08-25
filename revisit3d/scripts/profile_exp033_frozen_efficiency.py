#!/usr/bin/env python3
"""No-fit latency, memory, storage, and search audit for the frozen paper model."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from revisit3d.backbones import FrozenVGGTFeatures, FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.losses import relative_w2c_from_twist
from revisit3d.models import (
    SpatialPlasticityHead,
    backproject_tokens,
    build_geometry_head,
    local_knn_scale,
    visual_transport,
)
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256


def _stats(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.9)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
    }


def _cuda_profile(function, *, warmup: int, repetitions: int) -> dict:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    result = None
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        elapsed.append(float(start.elapsed_time(end)))
    del result
    torch.cuda.synchronize()
    return {
        **_stats(elapsed),
        "warmup": warmup,
        "repetitions": repetitions,
        "baseline_allocated_bytes": int(baseline),
        "incremental_peak_allocated_bytes": int(
            max(0, torch.cuda.max_memory_allocated() - baseline)
        ),
    }


def _cpu_profile(function, *, warmup: int, repetitions: int) -> dict:
    for _ in range(warmup):
        function()
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {**_stats(elapsed), "warmup": warmup, "repetitions": repetitions}


def _release(module: torch.nn.Module) -> None:
    module.cpu()
    del module
    gc.collect()
    torch.cuda.empty_cache()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/EXP-033_frozen_efficiency_audit_v10.yaml"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-033 requires CUDA")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    atom_path = Path(config["model"]["atom_checkpoint"])
    address_path = Path(config["model"]["address_artifact"])
    hashes = {
        "atom": _sha256(atom_path),
        "address": _sha256(address_path),
    }
    hash_checks = {
        "atom": hashes["atom"] == config["model"]["atom_sha256"],
        "address": hashes["address"] == config["model"]["address_sha256"],
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"frozen artifact hash mismatch: {hash_checks}")

    atom_checkpoint = torch.load(atom_path, map_location="cpu", weights_only=False)
    address = joblib.load(address_path)
    cache = torch.load(
        config["data"]["geometry_cache"],
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    payload = cache["rows"][0]["segments"]["a_prime_context"]
    head = SpatialPlasticityHead(
        feature_dim=int(config["foundation"]["feature_dim"])
    ).to(device)
    head.load_state_dict(atom_checkpoint["head"])
    head.eval().requires_grad_(False)
    current = CachedAtomSegment.from_cache(payload, "current", device)

    with torch.enable_grad():
        current_zero = current.atom(head)
        current_code = adapt_minimal(
            head,
            current,
            current_zero.code,
            step_size=float(config["method"]["step_size"]),
        )
    source_atom = replace(current_zero, code=current_code.detach())
    descriptor = current_zero.key.mean((1, 2))[0].detach().cpu()

    profiling = config["profiling"]
    method_warmup = int(profiling["method_warmup"])
    method_repetitions = int(profiling["method_repetitions"])

    def atom_build():
        with torch.no_grad():
            return current.atom(head)

    def current_ttt():
        with torch.enable_grad():
            return adapt_minimal(
                head,
                current,
                current_zero.code,
                step_size=float(config["method"]["step_size"]),
            )

    def transport():
        with torch.no_grad():
            return visual_transport(source_atom, current_zero).code

    def current_readout():
        with torch.no_grad():
            return head.depth(current.features, current.base_depth, current_code)

    strength = float(config["method"]["reuse_strength"])

    def current_only_total():
        with torch.enable_grad():
            zero = current.atom(head)
            code = adapt_minimal(
                head,
                current,
                zero.code,
                step_size=float(config["method"]["step_size"]),
            )
        with torch.no_grad():
            return head.depth(current.features, current.base_depth, code)

    def full_memory_total():
        with torch.enable_grad():
            zero = current.atom(head)
            code = adapt_minimal(
                head,
                current,
                zero.code,
                step_size=float(config["method"]["step_size"]),
            )
        with torch.no_grad():
            reused = (code + strength * visual_transport(source_atom, zero).code).clamp(
                -1, 1
            )
            return head.depth(current.features, current.base_depth, reused)

    method_timings = {
        "atom_construction": _cuda_profile(
            atom_build, warmup=method_warmup, repetitions=method_repetitions
        ),
        "one_local_ttt_step": _cuda_profile(
            current_ttt, warmup=method_warmup, repetitions=method_repetitions
        ),
        "visual_transport_one_record": _cuda_profile(
            transport, warmup=method_warmup, repetitions=method_repetitions
        ),
        "depth_readout": _cuda_profile(
            current_readout, warmup=method_warmup, repetitions=method_repetitions
        ),
        "current_only_total_after_foundation": _cuda_profile(
            current_only_total,
            warmup=method_warmup,
            repetitions=method_repetitions,
        ),
        "full_memory_total_after_foundation_excluding_address": _cuda_profile(
            full_memory_total,
            warmup=method_warmup,
            repetitions=method_repetitions,
        ),
    }

    atom_fields = {
        name: _tensor_bytes(getattr(source_atom, name).detach().cpu())
        for name in ("xyz", "scale", "key", "code", "confidence")
    }
    atom_fields["descriptor"] = _tensor_bytes(descriptor)
    record_tensor_bytes = sum(atom_fields.values())
    capacity = int(config["bank"]["capacity"])
    storage = {
        "record_tensor_payload_bytes": record_tensor_bytes,
        "record_tensor_payload_mib": record_tensor_bytes / 2**20,
        "field_bytes": atom_fields,
        "capacity": capacity,
        "capacity_tensor_payload_bytes": record_tensor_bytes * capacity,
        "capacity_tensor_payload_mib": record_tensor_bytes * capacity / 2**20,
        "python_container_overhead_included": False,
    }

    compiled = address["compiled_mips"]
    current_descriptor = descriptor.numpy().astype(np.float64)
    rng = np.random.default_rng(int(config["seed"]))
    address_timings = {}
    maximum_error = 0.0
    for size in config["bank"]["scaling_sizes"]:
        size = int(size)
        sources = rng.normal(size=(size, current_descriptor.size)).astype(np.float64)
        sources /= np.linalg.norm(sources, axis=1, keepdims=True).clip(1e-12)
        query = compiled["source"] + current_descriptor * compiled["interaction"]
        constant = compiled["intercept"] + current_descriptor @ compiled["current"]

        def vectorized_address():
            return int(np.argmax(constant + sources @ query))

        vectorized = constant + sources @ query
        scalar = np.asarray(
            [
                compiled["intercept"]
                + current_descriptor @ compiled["current"]
                + source
                @ (compiled["source"] + current_descriptor * compiled["interaction"])
                for source in sources
            ]
        )
        maximum_error = max(maximum_error, float(np.max(np.abs(vectorized - scalar))))
        address_timings[str(size)] = _cpu_profile(
            vectorized_address,
            warmup=int(profiling["address_warmup"]),
            repetitions=int(profiling["address_repetitions"]),
        )

    custom_parameters = sum(parameter.numel() for parameter in head.parameters())
    address_parameters = sum(
        np.asarray(compiled[name]).size
        for name in ("current", "source", "interaction")
    ) + 1
    parameter_summary = {
        "plasticity_head_parameters": custom_parameters,
        "factorized_address_parameters": int(address_parameters),
        "total_learned_addition_parameters": int(
            custom_parameters + address_parameters
        ),
        "plasticity_head_fp32_mib": custom_parameters * 4 / 2**20,
    }

    # Profile frozen foundation modules on the corresponding raw development context.
    dataset = RevisitEpisodeDataset(
        config["data"]["manifest"],
        config["data"]["scene_root"],
        split=config["data"]["split"],
        image_size=(
            int(config["data"]["image_height"]),
            int(config["data"]["image_width"]),
        ),
    )
    images = dataset[0]["a_prime"]["context"]["rgb"].unsqueeze(0).to(device)
    intrinsics = (
        dataset[0]["a_prime"]["context"]["intrinsics"].unsqueeze(0).to(device)
    )
    geometry_checkpoint = torch.load(
        config["geometry_head_checkpoint"], map_location="cpu", weights_only=False
    )
    extractor = FrozenVGGTFeatures(
        config["foundation"]["checkpoint"],
        repo_root=config["foundation"]["repository"],
    ).to(device)
    geometry_head = build_geometry_head(
        geometry_checkpoint["head_type"], extractor.feature_dim
    ).to(device)
    geometry_head.load_state_dict(geometry_checkpoint["head"])
    geometry_head.eval().requires_grad_(False)

    def geometry_foundation_forward():
        with torch.no_grad():
            features = extractor(images)
            state = geometry_head.initial_state(
                1, device=device, dtype=features.dtype
            )
            prediction = geometry_head(features, state)
            side = int(math.sqrt(prediction["depth"].shape[2]))
            depth = prediction["depth"].squeeze(-1).reshape(
                1, prediction["depth"].shape[1], side, side
            )
            w2c = relative_w2c_from_twist(prediction["relative_pose"])
            xyz = backproject_tokens(
                depth, intrinsics, w2c, image_size=tuple(images.shape[-2:])
            )
            scale = local_knn_scale(xyz)
            return features, prediction, xyz, scale

    foundation_timings = {
        "feature_geometry_pass": _cuda_profile(
            geometry_foundation_forward,
            warmup=int(profiling["foundation_warmup"]),
            repetitions=int(profiling["foundation_repetitions"]),
        )
    }
    _release(extractor)
    _release(geometry_head)

    tracker = FrozenVGGTGeometryTracker(
        config["foundation"]["checkpoint"],
        repo_root=config["foundation"]["repository"],
    ).to(device)
    queries = query_grid(
        images.shape[-2],
        images.shape[-1],
        int(config["foundation"]["track_side"]),
        str(device),
    )

    def tracker_forward():
        with torch.no_grad():
            return tracker(images, queries)

    foundation_timings["geometry_tracker_pass"] = _cuda_profile(
        tracker_forward,
        warmup=int(profiling["foundation_warmup"]),
        repetitions=int(profiling["foundation_repetitions"]),
    )
    _release(tracker)

    finite_timings = all(
        np.isfinite(value[key])
        for section in (method_timings, foundation_timings, address_timings)
        for value in section.values()
        for key in ("median_ms", "p90_ms", "mean_ms")
    )
    checks = {
        "frozen_artifact_hashes_match": all(hash_checks.values()),
        "all_timings_finite": bool(finite_timings),
        "address_factorization_error_at_most_1e-10": maximum_error <= 1e-10,
        "capacity_64_storage_reported": capacity == 64
        and storage["capacity_tensor_payload_bytes"] > 0,
    }
    result = {
        "experiment": "EXP-033",
        "stage": "frozen_efficiency_and_complexity_audit",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "split": config["data"]["split"],
        "input": {"views": 8, "height": 224, "width": 224},
        "artifact_hashes": hashes,
        "artifact_hash_checks": hash_checks,
        "method_timings": method_timings,
        "foundation_timings": foundation_timings,
        "address_timings": address_timings,
        "address_factorization_maximum_absolute_error": maximum_error,
        "storage": storage,
        "parameters": parameter_summary,
        "timing_scope": {
            "cuda_excludes": ["disk_io", "host_to_device_copy", "model_loading"],
            "foundation_implementation": (
                "separate FastVGGT feature/geometry and tracker passes"
            ),
            "address_device": "CPU float64 NumPy vectorized exact MIPS",
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "model_changed": False,
        "terminal_data_accessed": False,
    }
    output = Path(config["output"]["result"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
