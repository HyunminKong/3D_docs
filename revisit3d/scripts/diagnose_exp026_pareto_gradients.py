#!/usr/bin/env python3
"""No-fit audit of metric-gradient conflict in local plasticity meta-learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.models import visual_transport
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.scripts.train_exp012_minimal_atom import _component_folds, _new_head
from revisit3d.scripts.train_exp012_utility_selected_atom import _episode_segments
from revisit3d.scripts.train_exp024_metric_aligned_atom import _lidar_cache, _query_depth


def _metric_pair(prediction, target, valid, config):
    log_rows, relative_rows = [], []
    minimum_cells = int(config["lidar"]["minimum_cells_per_view"])
    objective = config.get("diagnosis", config.get("meta_objective"))
    if objective is None:
        raise KeyError("diagnosis or meta_objective settings are required")
    epsilon = float(objective["minimum_depth"])
    for view in range(prediction.shape[0]):
        mask = valid[view]
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].clamp_min(epsilon)
        gt = target[view][mask].clamp_min(epsilon)
        scale = torch.median(gt / pred).detach()
        aligned = (pred * scale).clamp_min(epsilon)
        log_rows.append((torch.log(aligned) - torch.log(gt)).abs().mean())
        relative_rows.append(((aligned - gt).abs() / gt).mean())
    if not log_rows:
        raise RuntimeError("query LiDAR has no valid view")
    return torch.stack(log_rows).mean(), torch.stack(relative_rows).mean()


def _objective_pair(head, sources, current, query, lidar, config):
    zero = current.atom(head)
    query_zero = query.atom(head)
    current_code = adapt_minimal(
        head, current, zero.code, step_size=float(config["method"]["step_size"])
    )
    current_atom = type(zero)(
        zero.xyz, zero.scale, zero.key, current_code, zero.confidence
    )
    current_log, current_relative = _metric_pair(
        _query_depth(head, query, query_zero, current_atom), *lidar, config
    )
    candidate_log, candidate_relative = [], []
    for source in sources:
        source_zero = source.atom(head)
        source_code = adapt_minimal(
            head, source, source_zero.code,
            step_size=float(config["method"]["step_size"]),
        )
        source_atom = type(source_zero)(
            source_zero.xyz, source_zero.scale, source_zero.key,
            source_code, source_zero.confidence,
        )
        transported = visual_transport(source_atom, zero).code
        reused_code = (
            current_code + float(config["method"]["reuse_strength"]) * transported
        ).clamp(-1, 1)
        reused_atom = type(zero)(
            zero.xyz, zero.scale, zero.key, reused_code, zero.confidence
        )
        log_loss, relative_loss = _metric_pair(
            _query_depth(head, query, query_zero, reused_atom), *lidar, config
        )
        candidate_log.append(log_loss)
        candidate_relative.append(relative_loss)
    candidate_log = torch.stack(candidate_log)
    candidate_relative = torch.stack(candidate_relative)
    best_log = candidate_log[int(candidate_log.detach().argmin())]
    best_relative = candidate_relative[int(candidate_relative.detach().argmin())]
    return (
        0.5 * (current_log + best_log),
        0.5 * (current_relative + best_relative),
        current_log,
        current_relative,
    )


def _flat_gradient(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _gradient_geometry(log_gradient, relative_gradient, config):
    epsilon = float(config["diagnosis"]["minimum_gradient_norm"])
    tolerance = float(config["diagnosis"]["descent_tolerance"])
    log_norm = log_gradient.norm()
    relative_norm = relative_gradient.norm()
    valid = bool(log_norm > epsilon and relative_norm > epsilon)
    if not valid:
        return {"valid": False, "log_norm": float(log_norm), "relative_norm": float(relative_norm)}
    log_unit = log_gradient / log_norm
    relative_unit = relative_gradient / relative_norm
    cosine = torch.dot(log_unit, relative_unit).clamp(-1, 1)

    raw_average = 0.5 * (log_gradient + relative_gradient)
    raw_norm = raw_average.norm().clamp_min(epsilon)
    raw_log_margin = torch.dot(log_unit, raw_average / raw_norm)
    raw_relative_margin = torch.dot(relative_unit, raw_average / raw_norm)

    bisector = log_unit + relative_unit
    bisector_norm = bisector.norm()
    if bisector_norm > epsilon:
        bisector_direction = bisector / bisector_norm
        bisector_log_margin = torch.dot(log_unit, bisector_direction)
        bisector_relative_margin = torch.dot(relative_unit, bisector_direction)
    else:
        bisector_log_margin = bisector.new_tensor(0.0)
        bisector_relative_margin = bisector.new_tensor(0.0)

    difference = log_gradient - relative_gradient
    denominator = torch.dot(difference, difference).clamp_min(epsilon)
    mixing = torch.clamp(
        (relative_norm.square() - torch.dot(log_gradient, relative_gradient)) / denominator,
        0.0, 1.0,
    )
    mgda = mixing * log_gradient + (1.0 - mixing) * relative_gradient
    mgda_norm = mgda.norm().clamp_min(epsilon)
    mgda_log_margin = torch.dot(log_unit, mgda / mgda_norm)
    mgda_relative_margin = torch.dot(relative_unit, mgda / mgda_norm)

    return {
        "valid": True,
        "log_norm": float(log_norm),
        "relative_norm": float(relative_norm),
        "norm_ratio_log_over_relative": float(log_norm / relative_norm),
        "cosine": float(cosine),
        "conflict": bool(cosine < 0),
        "raw_average_log_margin": float(raw_log_margin),
        "raw_average_relative_margin": float(raw_relative_margin),
        "raw_average_sacrifices_objective": bool(
            raw_log_margin <= tolerance or raw_relative_margin <= tolerance
        ),
        "bisector_norm_ratio": float(bisector_norm / 2.0),
        "bisector_log_margin": float(bisector_log_margin),
        "bisector_relative_margin": float(bisector_relative_margin),
        "bisector_common_descent": bool(
            bisector_log_margin > tolerance and bisector_relative_margin > tolerance
        ),
        "raw_mgda_mixing_log": float(mixing),
        "raw_mgda_log_margin": float(mgda_log_margin),
        "raw_mgda_relative_margin": float(mgda_relative_margin),
        "raw_mgda_common_descent": bool(
            mgda_log_margin > tolerance and mgda_relative_margin > tolerance
        ),
    }


def _load_anchor(anchor, cache, config, device):
    head = _new_head(cache, config, device, int(anchor.get("seed", config["seed"])))
    if anchor["kind"] == "checkpoint":
        payload = torch.load(anchor["path"], map_location=device, weights_only=False)
        head.load_state_dict(payload["head"], strict=True)
    elif anchor["kind"] != "fresh":
        raise ValueError(f"unknown anchor kind: {anchor['kind']}")
    head.eval()
    return head


def _component_balanced(rows, key):
    components = sorted({row["component"] for row in rows})
    values = [
        np.mean([float(row[key]) for row in rows if row["component"] == component])
        for component in components
    ]
    return float(np.mean(values))


def _anchor_summary(rows):
    valid = [row for row in rows if row["valid"]]
    return {
        "targets": len(rows),
        "valid_targets": len(valid),
        "components": len({row["component"] for row in rows}),
        "component_balanced_conflict_rate": _component_balanced(valid, "conflict"),
        "component_balanced_raw_average_sacrifice_rate": _component_balanced(
            valid, "raw_average_sacrifices_objective"
        ),
        "component_balanced_bisector_common_descent_rate": _component_balanced(
            valid, "bisector_common_descent"
        ),
        "component_balanced_raw_mgda_common_descent_rate": _component_balanced(
            valid, "raw_mgda_common_descent"
        ),
        "median_cosine": float(np.median([row["cosine"] for row in valid])),
        "median_norm_ratio_log_over_relative": float(np.median([
            row["norm_ratio_log_over_relative"] for row in valid
        ])),
        "median_bisector_norm_ratio": float(np.median([
            row["bisector_norm_ratio"] for row in valid
        ])),
        "median_bisector_worst_margin": float(np.median([
            min(row["bisector_log_margin"], row["bisector_relative_margin"])
            for row in valid
        ])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-026_pareto_gradient_diagnosis_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    if result_path.exists():
        raise RuntimeError("EXP-026 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-026 requires train split and CUDA")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior.get("experiment") == "EXP-025"
        and prior.get("registered_gate", {}).get("passed") is False
        and prior.get("exp021_terminal_accessed") is False
        and "checkpoint" not in prior
    ):
        raise RuntimeError("EXP-025 failure/terminal-access contract changed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    cache = torch.load(
        config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True
    )
    if not (
        len(manifest) == len(cache.get("rows", [])) == 225
        and cache.get("split") == "train"
        and cache.get("protocol_revision") == "v1.5"
    ):
        raise RuntimeError("EXP-026 data contract failed")
    _, group_of = _component_folds(manifest, 5, int(config["seed"]))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    lidar = _lidar_cache(manifest, cache, config, device)
    anchors, all_rows = {}, []
    for anchor in config["diagnosis"]["anchors"]:
        head = _load_anchor(anchor, cache, config, device)
        parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
        rows = []
        for index in range(len(manifest)):
            sources, _, current, query = _episode_segments(
                cache, manifest, index, list(range(len(manifest))), config, device
            )
            log_objective, relative_objective, current_log, current_relative = _objective_pair(
                head, sources, current, query, lidar[index], config
            )
            log_gradient = _flat_gradient(log_objective, parameters, retain_graph=True)
            relative_gradient = _flat_gradient(relative_objective, parameters, retain_graph=False)
            row = {
                "anchor": anchor["name"],
                "index": index,
                "episode": manifest[index]["episode_id"],
                "component": group_of[index],
                "location": manifest[index]["location"],
                "log_objective": float(log_objective.detach()),
                "relative_objective": float(relative_objective.detach()),
                "current_log": float(current_log.detach()),
                "current_relative": float(current_relative.detach()),
                **_gradient_geometry(log_gradient, relative_gradient, config),
            }
            rows.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(manifest):
                print(json.dumps({
                    "anchor": anchor["name"], "completed": index + 1,
                    "total": len(manifest),
                }), flush=True)
        anchors[anchor["name"]] = _anchor_summary(rows)
        all_rows.extend(rows)
        del head
        torch.cuda.empty_cache()

    success = config["success"]
    checks = {
        "coverage": all(
            row["targets"] >= int(success["minimum_targets_per_anchor"])
            and row["components"] >= int(success["minimum_components_per_anchor"])
            and row["valid_targets"] == row["targets"]
            for row in anchors.values()
        ),
        "conflict_is_material": max(
            row["component_balanced_conflict_rate"] for row in anchors.values()
        ) >= float(success["minimum_conflict_rate_any_anchor"]),
        "raw_average_sacrifice_is_material": max(
            row["component_balanced_raw_average_sacrifice_rate"] for row in anchors.values()
        ) >= float(success["minimum_raw_average_sacrifice_rate_any_anchor"]),
        "normalized_common_descent_exists": all(
            row["component_balanced_bisector_common_descent_rate"]
            >= float(success["minimum_normalized_common_descent_rate_all_anchors"])
            for row in anchors.values()
        ),
        "common_direction_is_not_degenerate": all(
            row["median_bisector_norm_ratio"]
            >= float(success["minimum_median_bisector_norm_ratio_all_anchors"])
            for row in anchors.values()
        ),
    }
    result = {
        "experiment": "EXP-026",
        "stage": "no_fit_pareto_gradient_diagnosis",
        "protocol_revision": config["protocol_revision"],
        "config": str(config_path),
        "split": "train",
        "no_model_fit": True,
        "parameter_updates": 0,
        "online_loss": "track3d_only",
        "offline_objectives": list(config["diagnosis"]["objectives"]),
        "gradient_rule_audited": "unit_normalized_bisector",
        "anchors": anchors,
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "rows": all_rows,
        "validation_accessed": False,
        "test_accessed": False,
        "exp021_terminal_accessed": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "anchors": anchors,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
