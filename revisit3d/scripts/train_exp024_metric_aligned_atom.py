#!/usr/bin/env python3
"""Cross-fit one-loss metric-aligned local plasticity atom."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp010_absolute_geometry import LidarProjector, _depth_metrics, _query_lidar
from revisit3d.scripts.evaluate_exp023_metric_utility_oracle import _metric_risk
from revisit3d.scripts.train_exp012_minimal_atom import _component_folds, _new_head, _sha256
from revisit3d.scripts.train_exp012_utility_selected_atom import _episode_segments


METRICS = ("silog", "abs_rel", "rmse_m", "delta1", "point_epe_m")


def _lidar_cache(manifest, geometry, config, device):
    projector = LidarProjector(
        config["data"]["nuscenes_root"],
        minimum_depth=float(config["lidar"]["minimum_depth_m"]),
        maximum_depth=float(config["lidar"]["maximum_depth_m"]),
    )
    scene_root = Path(config["data"]["scene_root"])
    output = {}
    for index, row in enumerate(manifest):
        query_payload = geometry["rows"][index]["segments"]["a_prime_query"]
        side = int(query_payload["base_depth"].shape[-1])
        depth, valid = _query_lidar(projector, scene_root, row["a_prime"], side)
        output[index] = (
            torch.as_tensor(depth, dtype=torch.float32, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device),
        )
    return output


def _query_depth(head, query, query_zero, atom):
    code = visual_transport(atom, query_zero).code
    depth = head.depth(query.features, query.base_depth, code)
    points = depth.shape[2]
    side = int(points ** 0.5)
    return depth[0, :, :, 0].reshape(-1, side, side)


def _metric_loss(prediction, target, valid, config):
    rows = []
    minimum_cells = int(config["lidar"]["minimum_cells_per_view"])
    epsilon = float(config["meta_objective"]["minimum_depth"])
    for view in range(prediction.shape[0]):
        mask = valid[view]
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].clamp_min(epsilon)
        gt = target[view][mask].clamp_min(epsilon)
        scale = torch.median(gt / pred).detach()
        rows.append((torch.log(pred * scale) - torch.log(gt)).abs().mean())
    if not rows:
        raise RuntimeError("query LiDAR has no valid view")
    return torch.stack(rows).mean()


def _rollout(head, sources, current, query, lidar, config):
    zero = current.atom(head)
    query_zero = query.atom(head)
    current_code = adapt_minimal(
        head, current, zero.code, step_size=float(config["method"]["step_size"])
    )
    current_atom = replace(zero, code=current_code)
    current_depth = _query_depth(head, query, query_zero, current_atom)
    current_loss = _metric_loss(current_depth, *lidar, config)
    candidate_losses = []
    candidate_atoms = []
    for source in sources:
        source_zero = source.atom(head)
        source_code = adapt_minimal(
            head, source, source_zero.code, step_size=float(config["method"]["step_size"])
        )
        source_atom = replace(source_zero, code=source_code)
        transported = visual_transport(source_atom, zero).code
        candidate_code = (
            current_code + float(config["method"]["reuse_strength"]) * transported
        ).clamp(-1, 1)
        candidate_atom = replace(zero, code=candidate_code)
        candidate_atoms.append(candidate_atom)
        candidate_losses.append(
            _metric_loss(_query_depth(head, query, query_zero, candidate_atom), *lidar, config)
        )
    candidate_losses = torch.stack(candidate_losses)
    selected = int(candidate_losses.detach().argmin())
    outer = 0.5 * (current_loss + candidate_losses[selected])
    return outer, current_loss, candidate_losses, current_atom, candidate_atoms, query_zero


def _train(head, geometry, manifest, lidar, indices, config, device, *, seed, label):
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    generator = random.Random(seed)
    order = []
    logs = []
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
        outer, current_loss, candidates, _, _, _ = _rollout(
            head, sources, current, query, lidar[index], config
        )
        if not torch.isfinite(outer):
            raise RuntimeError(f"non-finite metric outer loss at {label}:{step}")
        outer.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters, float(config["training"]["gradient_clip"])
        )
        optimizer.step()
        if step == 1 or step % int(config["training"]["log_every"]) == 0 or step == int(config["training"]["steps"]):
            selected = int(candidates.detach().argmin())
            logs.append({
                "step": step,
                "outer": float(outer.detach()),
                "current_metric_loss": float(current_loss.detach()),
                "best_candidate_metric_loss": float(candidates[selected].detach()),
                "selected_label": labels[selected],
                "gradient_norm": float(gradient),
            })
            print(json.dumps({"phase": label, **logs[-1]}), flush=True)
    return logs


def _numpy_metrics(prediction, target, valid, query, config):
    return _depth_metrics(
        prediction.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
        query.intrinsics[0].detach().cpu().numpy(),
        image_size=query.image_size,
        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
    )


def _risk(prediction, target, valid, config):
    return _metric_risk(
        prediction.detach().cpu().numpy(),
        target.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
        minimum_cells=int(config["lidar"]["minimum_cells_per_view"]),
        minimum_depth=float(config["meta_objective"]["minimum_depth"]),
    )


def _evaluate(head, geometry, manifest, lidar, indices, pool, group_of, config, device):
    rows = []
    head.eval()
    with torch.enable_grad():
        for index in indices:
            sources, labels, current, query = _episode_segments(
                geometry, manifest, index, pool, config, device
            )
            _, _, _, current_atom, candidate_atoms, query_zero = _rollout(
                head, sources, current, query, lidar[index], config
            )
            target, valid = lidar[index]
            base_prediction = query.base_depth[0]
            current_prediction = _query_depth(head, query, query_zero, current_atom)
            candidate_predictions = [
                _query_depth(head, query, query_zero, atom) for atom in candidate_atoms
            ]
            base = {
                "metric_risk": _risk(base_prediction, target, valid, config),
                "metrics": _numpy_metrics(base_prediction, target, valid, query, config),
            }
            current_row = {
                "metric_risk": _risk(current_prediction, target, valid, config),
                "metrics": _numpy_metrics(current_prediction, target, valid, query, config),
            }
            candidates = [
                {
                    "label": label,
                    "metric_risk": _risk(prediction, target, valid, config),
                    "metrics": _numpy_metrics(prediction, target, valid, query, config),
                }
                for label, prediction in zip(labels, candidate_predictions)
            ]
            oracle = min(candidates, key=lambda row: row["metric_risk"])
            random_row = {
                "metric_risk": float(np.mean([row["metric_risk"] for row in candidates])),
                "metrics": {
                    metric: float(np.mean([row["metrics"][metric] for row in candidates]))
                    for metric in METRICS
                },
            }
            denominator = max(abs(current_row["metric_risk"]), 1e-8)
            utilities = [
                (current_row["metric_risk"] - row["metric_risk"]) / denominator
                for row in candidates
            ]
            rows.append({
                "index": index,
                "episode": manifest[index]["episode_id"],
                "component": group_of[index],
                "location": manifest[index]["location"],
                "base": base,
                "current": current_row,
                "metric_oracle": oracle,
                "random_expectation": random_row,
                "candidate_metric_utilities": utilities,
            })
    return rows


def _summary(rows, policy):
    components = sorted({row["component"] for row in rows})
    return {
        "targets": len(rows),
        "components": len(components),
        "metric_risk": float(np.mean([
            np.mean([row[policy]["metric_risk"] for row in rows if row["component"] == component])
            for component in components
        ])),
        **{
            metric: float(np.mean([
                np.mean([row[policy]["metrics"][metric] for row in rows if row["component"] == component])
                for component in components
            ]))
            for metric in METRICS
        },
    }


def _bootstrap(rows, left, right, value, *, samples, seed):
    components = sorted({row["component"] for row in rows})
    if value == "metric_risk":
        getter = lambda row, policy: row[policy]["metric_risk"]
    else:
        getter = lambda row, policy: row[policy]["metrics"][value]
    values = np.asarray([
        np.mean([
            getter(row, left) - getter(row, right)
            for row in rows if row["component"] == component
        ])
        for component in components
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "direction": f"{left}_minus_{right}_positive_means_{right}_better",
        "components": len(components),
        "mean_improvement": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-024_metric_aligned_atom_v10.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    result_path = Path(config["output"]["result"])
    checkpoint_path = Path(config["output"]["checkpoint"])
    if result_path.exists() or checkpoint_path.exists():
        raise RuntimeError("EXP-024 output already exists")
    if config["data"]["split"] != "train" or not torch.cuda.is_available():
        raise RuntimeError("EXP-024 requires train split and CUDA")
    prior = json.loads(Path(config["prior_stage"]).read_text())
    if not (
        prior["experiment"] == "EXP-023"
        and prior["registered_gate"]["passed"] is True
        and prior["no_model_fit"] is True
        and prior["exp021_terminal_accessed"] is False
    ):
        raise RuntimeError("EXP-023 authorization contract failed")
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
        raise RuntimeError("EXP-024 data/budget contract failed")
    folds, group_of = _component_folds(
        manifest, int(config["crossfit"]["folds"]), int(config["seed"])
    )
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    lidar = _lidar_cache(manifest, geometry, config, device)
    oof_rows = []
    fold_logs = []
    for fold, held_out in enumerate(folds):
        train_indices = [index for index in range(len(manifest)) if index not in held_out]
        head = _new_head(geometry, config, device, int(config["seed"]) + fold)
        logs = _train(
            head, geometry, manifest, lidar, train_indices, config, device,
            seed=int(config["seed"]) + fold, label=f"metric-fold-{fold}",
        )
        oof_rows.extend(
            _evaluate(
                head, geometry, manifest, lidar, held_out, held_out,
                group_of, config, device,
            )
        )
        fold_logs.append({
            "fold": fold, "train": len(train_indices), "held_out": len(held_out), "logs": logs
        })
    if sorted(row["index"] for row in oof_rows) != list(range(len(manifest))):
        raise RuntimeError("EXP-024 OOF partition changed")
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
            metric: _bootstrap(oof_rows, "current", "metric_oracle", metric, samples=samples, seed=seed + 10 + i)
            for i, metric in enumerate(primary)
        },
        "oracle_vs_random_risk": _bootstrap(
            oof_rows, "random_expectation", "metric_oracle", "metric_risk",
            samples=samples, seed=seed + 20,
        ),
    }
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
    }
    utilities = np.asarray([
        value for row in oof_rows for value in row["candidate_metric_utilities"]
    ], dtype=np.float64)
    base_result = {
        "experiment": "EXP-024",
        "stage": "metric_aligned_atom_crossfit",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "config": str(config_path),
        "online_loss": "track3d_only",
        "offline_meta_loss": config["meta_objective"]["loss"],
        "meta_aggregation": config["meta_objective"]["aggregation"],
        "auxiliary_losses": [],
        "training_steps": int(config["training"]["steps"]),
        "summaries": summaries,
        "comparisons": comparisons,
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
            "comparisons": comparisons, "gate": base_result["registered_gate"],
        }), flush=True)
        return
    final_head = _new_head(geometry, config, device, int(config["seed"]))
    final_logs = _train(
        final_head, geometry, manifest, lidar, list(range(len(manifest))), config, device,
        seed=int(config["seed"]), label="metric-full-train",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "EXP-024",
        "stage": "metric_aligned_atom",
        "protocol_revision": config["protocol_revision"],
        "split": "train",
        "head": final_head.state_dict(),
        "visual_key": "frozen_train_pca",
        "online_loss": "track3d_only",
        "offline_meta_loss": config["meta_objective"]["loss"],
        "auxiliary_losses": [],
        "training_steps": int(config["training"]["steps"]),
        "step_size": float(config["method"]["step_size"]),
        "reuse_strength": float(config["method"]["reuse_strength"]),
        "validation_accessed": False,
        "test_accessed": False,
        "exp021_terminal_accessed": False,
    }, checkpoint_path)
    result = {
        **base_result,
        "final_logs": final_logs,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({
        "output": str(result_path), "checkpoint": str(checkpoint_path),
        "summaries": summaries, "comparisons": comparisons,
        "candidate_metric_utility": result["candidate_metric_utility"],
        "gate": result["registered_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
