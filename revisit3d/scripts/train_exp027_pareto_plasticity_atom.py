#!/usr/bin/env python3
"""Coefficient-free two-objective plasticity-head meta-training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.scripts.diagnose_exp026_pareto_gradients import (
    _flat_gradient,
    _objective_pair,
)
from revisit3d.scripts.train_exp012_minimal_atom import (
    _component_folds,
    _new_head,
    _sha256,
)
from revisit3d.scripts.train_exp012_utility_selected_atom import _episode_segments
from revisit3d.scripts.train_exp024_metric_aligned_atom import (
    METRICS,
    _bootstrap,
    _evaluate,
    _lidar_cache,
    _summary,
)


def _assign_flat_gradient(parameters, gradient):
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[offset:offset + count].reshape_as(parameter).clone()
        offset += count
    if offset != gradient.numel():
        raise RuntimeError("flat gradient does not match trainable parameters")


def _flat_parameters(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def _optimizer_step(
    parameters, optimizer, common_gradient, log_unit, relative_unit, *, epsilon, tolerance
):
    """EXP-027 reference: accept the AdamW displacement without safeguarding."""
    _assign_flat_gradient(parameters, common_gradient)
    before = _flat_parameters(parameters)
    optimizer.step()
    displacement = _flat_parameters(parameters) - before
    descent = -displacement
    descent_norm = descent.norm().clamp_min(epsilon)
    realized_log_margin = torch.dot(log_unit, descent / descent_norm)
    realized_relative_margin = torch.dot(relative_unit, descent / descent_norm)
    return {
        "realized_log_margin": realized_log_margin,
        "realized_relative_margin": realized_relative_margin,
        "safeguard_applied": False,
    }


def _train(head, geometry, manifest, lidar, indices, group_of, config, device, *, seed, label):
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = random.Random(seed)
    order, logs, displacement_rows = [], [], []
    epsilon = float(config["meta_objective"]["minimum_gradient_norm"])
    tolerance = float(config["meta_objective"]["descent_tolerance"])
    head.train()
    for step in range(1, int(config["training"]["steps"]) + 1):
        if not order:
            order = list(indices)
            generator.shuffle(order)
        index = order.pop()
        sources, labels, current, query = _episode_segments(
            geometry, manifest, index, indices, config, device
        )
        optimizer.zero_grad(set_to_none=True)
        log_objective, relative_objective, current_log, current_relative = _objective_pair(
            head, sources, current, query, lidar[index], config
        )
        log_gradient = _flat_gradient(log_objective, parameters, retain_graph=True)
        relative_gradient = _flat_gradient(relative_objective, parameters, retain_graph=False)
        log_norm = log_gradient.norm()
        relative_norm = relative_gradient.norm()
        if not (torch.isfinite(log_norm) and torch.isfinite(relative_norm)):
            raise RuntimeError(f"non-finite EXP-027 gradient at {label}:{step}")
        if log_norm <= epsilon or relative_norm <= epsilon:
            raise RuntimeError(f"degenerate EXP-027 gradient at {label}:{step}")
        log_unit = log_gradient / log_norm
        relative_unit = relative_gradient / relative_norm
        common_gradient = 0.5 * (log_unit + relative_unit)
        synthesis_log_margin = torch.dot(log_unit, common_gradient)
        synthesis_relative_margin = torch.dot(relative_unit, common_gradient)
        if synthesis_log_margin <= tolerance or synthesis_relative_margin <= tolerance:
            raise RuntimeError(f"EXP-027 synthesis lost common descent at {label}:{step}")
        # The arithmetic mean of two unit gradients has norm at most one, so
        # the registered clip bound is already satisfied exactly.
        clipped_norm = common_gradient.norm()
        step_result = _optimizer_step(
            parameters, optimizer, common_gradient, log_unit, relative_unit,
            epsilon=epsilon, tolerance=tolerance,
        )
        realized_log_margin = step_result["realized_log_margin"]
        realized_relative_margin = step_result["realized_relative_margin"]
        displacement_rows.append({
            "component": group_of[index],
            "synthesized_common_descent": True,
            "realized_common_descent": bool(
                realized_log_margin > tolerance and realized_relative_margin > tolerance
            ),
            "realized_log_margin": float(realized_log_margin),
            "realized_relative_margin": float(realized_relative_margin),
            "safeguard_applied": bool(step_result["safeguard_applied"]),
        })
        if (
            step == 1
            or step % int(config["training"]["log_every"]) == 0
            or step == int(config["training"]["steps"])
        ):
            row = {
                "step": step,
                "log_objective": float(log_objective.detach()),
                "relative_objective": float(relative_objective.detach()),
                "current_log": float(current_log.detach()),
                "current_relative": float(current_relative.detach()),
                "gradient_cosine": float(torch.dot(log_unit, relative_unit)),
                "synthesis_worst_margin": float(torch.minimum(
                    synthesis_log_margin, synthesis_relative_margin
                )),
                "realized_worst_margin": float(torch.minimum(
                    realized_log_margin, realized_relative_margin
                )),
                "gradient_norm": float(clipped_norm),
            }
            logs.append(row)
            print(json.dumps({"phase": label, **row}), flush=True)
    return logs, displacement_rows


def _displacement_summary(rows):
    components = sorted({row["component"] for row in rows})
    component_rates = [
        np.mean([
            row["realized_common_descent"]
            for row in rows if row["component"] == component
        ])
        for component in components
    ]
    return {
        "steps": len(rows),
        "components": len(components),
        "synthesized_common_descent_rate": float(np.mean([
            row["synthesized_common_descent"] for row in rows
        ])),
        "realized_common_descent_rate": float(np.mean([
            row["realized_common_descent"] for row in rows
        ])),
        "component_balanced_realized_common_descent_rate": float(np.mean(component_rates)),
        "safeguard_rate": float(np.mean([row["safeguard_applied"] for row in rows])),
        "median_realized_worst_margin": float(np.median([
            min(row["realized_log_margin"], row["realized_relative_margin"])
            for row in rows
        ])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-027_pareto_plasticity_atom_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError("EXP-027 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-027 requires train split and CUDA")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    experiment = config["experiment"]
    if experiment == "EXP-027":
        authorized = (
            prior.get("experiment") == "EXP-026"
            and prior.get("no_model_fit") is True
            and prior.get("parameter_updates") == 0
            and prior.get("registered_gate", {}).get("passed") is True
            and prior.get("exp021_terminal_accessed") is False
        )
    elif experiment == "EXP-028":
        checks = prior.get("registered_gate", {}).get("checks", {})
        authorized = (
            prior.get("experiment") == "EXP-027"
            and prior.get("registered_gate", {}).get("passed") is False
            and checks.get("current_all_metric_means_improve") is False
            and checks.get("realized_optimizer_common_descent") is False
            and all(checks.get(key) is True for key in (
                "coverage", "positive_current_intervals",
                "oracle_all_metric_means_improve", "positive_oracle_intervals",
                "oracle_risk_interval_over_random",
            ))
            and prior.get("exp021_terminal_accessed") is False
            and "checkpoint" not in prior
        )
    else:
        authorized = False
    if not authorized:
        raise RuntimeError(f"{experiment} authorization contract failed")
    manifest = json.loads(Path(config["data"]["manifest"]).read_text())
    geometry = torch.load(
        config["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True
    )
    if not (
        len(manifest) == len(geometry.get("rows", [])) == 225
        and geometry.get("split") == "train"
        and geometry.get("protocol_revision") == "v1.5"
        and int(config["training"]["steps"]) == 1000
    ):
        raise RuntimeError("EXP-027 data/budget contract failed")
    folds, group_of = _component_folds(
        manifest, int(config["crossfit"]["folds"]), int(config["seed"])
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    lidar = _lidar_cache(manifest, geometry, config, device)
    oof_rows, fold_logs, displacement_rows = [], [], []
    for fold, held_out in enumerate(folds):
        train_indices = [index for index in range(len(manifest)) if index not in held_out]
        head = _new_head(geometry, config, device, int(config["seed"]) + fold)
        logs, displacement = _train(
            head, geometry, manifest, lidar, train_indices, group_of, config, device,
            seed=int(config["seed"]) + fold, label=f"pareto-fold-{fold}",
        )
        oof_rows.extend(_evaluate(
            head, geometry, manifest, lidar, held_out, held_out,
            group_of, config, device,
        ))
        displacement_rows.extend(displacement)
        fold_logs.append({
            "fold": fold, "train": len(train_indices), "held_out": len(held_out),
            "logs": logs, "displacement": _displacement_summary(displacement),
        })
    if sorted(row["index"] for row in oof_rows) != list(range(len(manifest))):
        raise RuntimeError("EXP-027 OOF partition changed")
    policies = ("base", "current", "metric_oracle", "random_expectation")
    summaries = {policy: _summary(oof_rows, policy) for policy in policies}
    primary = tuple(config["success"]["primary_metrics"])
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    comparisons = {
        "current_vs_base": {
            metric: _bootstrap(oof_rows, "base", "current", metric, samples=samples, seed=seed + i)
            for i, metric in enumerate(primary)
        },
        "oracle_vs_current": {
            metric: _bootstrap(
                oof_rows, "current", "metric_oracle", metric,
                samples=samples, seed=seed + 10 + i,
            )
            for i, metric in enumerate(primary)
        },
        "oracle_vs_random_risk": _bootstrap(
            oof_rows, "random_expectation", "metric_oracle", "metric_risk",
            samples=samples, seed=seed + 20,
        ),
    }
    displacement = _displacement_summary(displacement_rows)
    checks = {
        "coverage": len(oof_rows) >= int(config["success"]["minimum_targets"])
        and len({row["component"] for row in oof_rows}) >= int(config["success"]["minimum_components"]),
        "current_all_metric_means_improve": all(
            summaries["current"][metric] < summaries["base"][metric] for metric in primary
        ),
        "positive_current_intervals": sum(
            comparisons["current_vs_base"][metric]["ci95"][0] > 0 for metric in primary
        ) >= int(config["success"]["minimum_positive_current_intervals"]),
        "oracle_all_metric_means_improve": all(
            summaries["metric_oracle"][metric] < summaries["current"][metric] for metric in primary
        ),
        "positive_oracle_intervals": sum(
            comparisons["oracle_vs_current"][metric]["ci95"][0] > 0 for metric in primary
        ) >= int(config["success"]["minimum_positive_oracle_intervals"]),
        "oracle_risk_interval_over_random": comparisons["oracle_vs_random_risk"]["ci95"][0] > 0,
        "realized_optimizer_common_descent":
            displacement["component_balanced_realized_common_descent_rate"]
            >= float(config["success"]["minimum_realized_common_descent_rate"]),
    }
    utilities = np.asarray([
        value for row in oof_rows for value in row["candidate_metric_utilities"]
    ], dtype=np.float64)
    base_result = {
        "experiment": experiment,
        "stage": config["output"].get("result_stage", "pareto_plasticity_atom_crossfit"),
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "config": str(config_path),
        "online_loss": "track3d_only",
        "offline_objectives": list(config["meta_objective"]["objectives"]),
        "gradient_synthesis": config["meta_objective"]["gradient_synthesis"],
        "loss_weights": [],
        "auxiliary_losses": [],
        "training_steps": int(config["training"]["steps"]),
        "summaries": summaries,
        "comparisons": comparisons,
        "optimizer_displacement": displacement,
        "candidate_metric_utility": {
            "mean": float(utilities.mean()),
            "harmful_rate": float(np.mean(utilities < 0)),
        },
        "registered_gate": {"checks": checks, "passed": all(checks.values())},
        "oof_rows": oof_rows,
        "fold_logs": fold_logs,
        "validation_accessed": False,
        "test_accessed": False,
        "exp021_terminal_accessed": False,
    }
    if not all(checks.values()):
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(base_result, indent=2, allow_nan=False))
        print(json.dumps({
            "output": str(result_path), "summaries": summaries,
            "comparisons": comparisons, "optimizer_displacement": displacement,
            "gate": base_result["registered_gate"],
        }), flush=True)
        return
    final_head = _new_head(geometry, config, device, int(config["seed"]))
    final_logs, final_displacement_rows = _train(
        final_head, geometry, manifest, lidar, list(range(len(manifest))),
        group_of, config, device, seed=int(config["seed"]), label="pareto-full-train",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": experiment,
        "stage": config["output"].get("checkpoint_stage", "pareto_plasticity_atom"),
        "protocol_revision": config["protocol_revision"], "split": "train",
        "head": final_head.state_dict(), "visual_key": "frozen_train_pca",
        "online_loss": "track3d_only",
        "offline_objectives": list(config["meta_objective"]["objectives"]),
        "gradient_synthesis": config["meta_objective"]["gradient_synthesis"],
        "loss_weights": [], "auxiliary_losses": [],
        "training_steps": int(config["training"]["steps"]),
        "step_size": float(config["method"]["step_size"]),
        "reuse_strength": float(config["method"]["reuse_strength"]),
        "validation_accessed": False, "test_accessed": False,
        "exp021_terminal_accessed": False,
    }, checkpoint_path)
    result = {
        **base_result,
        "final_logs": final_logs,
        "final_optimizer_displacement": _displacement_summary(final_displacement_rows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "checkpoint": str(checkpoint_path),
        "summaries": summaries, "comparisons": comparisons,
        "optimizer_displacement": displacement,
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
