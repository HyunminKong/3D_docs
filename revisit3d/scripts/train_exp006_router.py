#!/usr/bin/env python3
"""Train leakage-safe grouped-OOF neural utility/risk routers for EXP-006 v2.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from revisit3d.losses import utility_risk_loss
from revisit3d.models import ObservableUtilityRiskRouter


def _arrays(rows: list[dict], scalar_indices: list[int], device: torch.device) -> tuple[torch.Tensor, ...]:
    values = np.asarray([row["features"] for row in rows], dtype=np.float32)
    current = torch.from_numpy(values[:, :64]).to(device)
    candidate = torch.from_numpy(values[:, 64:128]).to(device)
    scalar_columns = 256 + np.asarray(scalar_indices, dtype=np.int64)
    scalars = torch.from_numpy(values[:, scalar_columns]).to(device)
    target = torch.tensor([row["future_utility"] for row in rows], device=device)
    return current, candidate, scalars, target


def _fit(
    train_rows: list[dict], *, scalar_indices: list[int], config: dict, seed: int, device: torch.device,
) -> tuple[ObservableUtilityRiskRouter, torch.Tensor, torch.Tensor, list[dict]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scalar_dim = len(scalar_indices)
    current, candidate, scalars, target = _arrays(train_rows, scalar_indices, device)
    if scalar_dim:
        scalar_mean = scalars.mean(dim=0)
        scalar_std = scalars.std(dim=0).clamp_min(1e-5)
    else:
        scalar_mean = scalars.new_empty(0)
        scalar_std = scalars.new_empty(0)
    scalars = (scalars - scalar_mean) / scalar_std
    stage2 = config["stage2"]
    model = ObservableUtilityRiskRouter(
        descriptor_dim=int(stage2["descriptor_dim"]), scalar_dim=scalar_dim,
        projected_dim=int(stage2["descriptor_projection_dim"]),
        hidden_dim=int(stage2["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(stage2["learning_rate"]),
        weight_decay=float(stage2["weight_decay"]),
    )
    epsilon = float(config["stage1"]["utility_deadband_minimum"])
    nonneutral = target.abs() > epsilon
    harmful = target < -epsilon
    positives = int((nonneutral & harmful).sum())
    negatives = int((nonneutral & ~harmful).sum())
    if positives == 0 or negatives == 0:
        raise RuntimeError("risk training partition must contain beneficial and harmful candidates")
    positive_weight = target.new_tensor(negatives / positives)
    logs = []
    model.train()
    for step in range(1, int(stage2["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current, candidate[:, None], scalars[:, None])
        utility = prediction.utility[:, 0]
        risk = prediction.risk_logit[:, 0]
        utility_loss = torch.nn.functional.smooth_l1_loss(
            utility, target, beta=float(stage2["utility_smooth_l1_beta"]),
        )
        _, risk_terms = utility_risk_loss(
            utility, risk, target, torch.ones_like(target, dtype=torch.bool),
            epsilon=epsilon, positive_weight=positive_weight,
        )
        loss = utility_loss + float(stage2["risk_loss_weight"]) * risk_terms["risk"]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(stage2["gradient_clip"]),
        )
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == int(stage2["steps"]):
            logs.append({
                "step": step, "loss": float(loss.detach()),
                "utility_loss": float(utility_loss.detach()),
                "risk_loss": float(risk_terms["risk"].detach()),
                "gradient_norm": float(gradient_norm),
            })
    return model, scalar_mean, scalar_std, logs


def _predict(
    model: ObservableUtilityRiskRouter,
    rows: list[dict], scalar_indices: list[int], scalar_mean: torch.Tensor, scalar_std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    current, candidate, scalars, _ = _arrays(rows, scalar_indices, device)
    model.eval()
    with torch.no_grad():
        prediction = model(
            current, candidate[:, None], ((scalars - scalar_mean) / scalar_std)[:, None],
        )
    return (
        prediction.utility[:, 0].cpu().numpy(),
        prediction.risk_logit[:, 0].sigmoid().cpu().numpy(),
    )


def _metrics(rows: list[dict], utility_hat: np.ndarray, risk: np.ndarray, epsilon: float) -> dict:
    target = np.asarray([row["future_utility"] for row in rows])
    selected, oracle, accepted = [], [], []
    for episode in sorted({row["episode"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["episode"] == episode]
        eligible = [index for index in indices if utility_hat[index] > 0 and risk[index] < 0.5]
        if eligible:
            choice = max(eligible, key=lambda index: utility_hat[index])
            value, accept = target[choice], True
        else:
            value, accept = 0.0, False
        selected.append(float(value))
        oracle.append(max(0.0, float(target[indices].max())))
        accepted.append(accept)
    selected_values = np.asarray(selected)
    nonneutral = np.abs(target) > epsilon
    risk_target = target < -epsilon
    risk_auc = None
    risk_ap = None
    if len(set(risk_target[nonneutral].tolist())) == 2:
        risk_auc = float(roc_auc_score(risk_target[nonneutral], risk[nonneutral]))
        risk_ap = float(average_precision_score(risk_target[nonneutral], risk[nonneutral]))
    return {
        "candidate_utility_spearman": float(spearmanr(utility_hat, target).statistic),
        "candidate_utility_mae": float(np.abs(utility_hat - target).mean()),
        "risk_auroc_nonneutral": risk_auc,
        "risk_average_precision_nonneutral": risk_ap,
        "episodes": len(selected),
        "mean_selected_utility": float(selected_values.mean()),
        "median_selected_utility": float(np.median(selected_values)),
        "beneficial_rate": float(np.mean(selected_values > epsilon)),
        "harmful_rate": float(np.mean(selected_values < -epsilon)),
        "accept_rate": float(np.mean(accepted)),
        "mean_oracle_utility": float(np.mean(oracle)),
        "mean_future_utility_regret": float(np.mean(np.asarray(oracle) - selected_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/EXP-006_utility_router_v27.yaml")
    parser.add_argument(
        "--out", default="revisit3d/results/EXP-006/stage2_neural_router_crossfit_expanded_train_v27.json",
    )
    parser.add_argument(
        "--checkpoint-dir", default="revisit3d/checkpoints/EXP-006_router_expanded_v27",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("EXP-006 router training requires CUDA")
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("protocol_revision") != "v2.7" or config["data"].get("split") != "train":
        raise RuntimeError("neural router requires the train-only v2.7 protocol")
    payload = json.loads(Path(config["stage2"]["source_features"]).read_text())
    if not (
        payload.get("protocol_revision") == "v2.7"
        and payload.get("validation_accessed") is False
        and payload.get("router_feature_contract", {}).get("query_or_future_input") is False
    ):
        raise RuntimeError("router features violate the v2.7 leakage boundary")
    rows = payload["router_features"]
    if any(len(row["features"]) != 280 for row in rows):
        raise RuntimeError("unexpected router feature dimension")
    device = torch.device("cuda")
    epsilon = float(payload["utility_epsilon"])
    variants = {
        "appearance_only": [],
        "appearance_online_history": [*range(12), *range(16, 24)],
        "appearance_online_geometry": [*range(24)],
    }
    requested = config["stage2"]["variants"]
    if any(name not in variants for name in requested):
        raise RuntimeError(f"unknown router variant in {requested}")
    folds = sorted({int(row["fold"]) for row in rows})
    results = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for variant in requested:
        scalar_indices = variants[variant]
        scalar_dim = len(scalar_indices)
        utility_hat = np.empty(len(rows), dtype=np.float64)
        risk_hat = np.empty(len(rows), dtype=np.float64)
        fold_records = []
        for fold in folds:
            train_indices = [index for index, row in enumerate(rows) if int(row["fold"]) != fold]
            test_indices = [index for index, row in enumerate(rows) if int(row["fold"]) == fold]
            model, mean, std, logs = _fit(
                [rows[index] for index in train_indices], scalar_indices=scalar_indices,
                config=config, seed=int(config["seed"]) + fold, device=device,
            )
            utility, risk = _predict(
                model, [rows[index] for index in test_indices], scalar_indices, mean, std, device,
            )
            utility_hat[test_indices], risk_hat[test_indices] = utility, risk
            fold_records.append({
                "fold": fold, "train_candidates": len(train_indices),
                "held_out_candidates": len(test_indices), "logs": logs,
            })
        metrics = _metrics(rows, utility_hat, risk_hat, epsilon)
        final_model, final_mean, final_std, final_logs = _fit(
            rows, scalar_indices=scalar_indices, config=config,
            seed=int(config["seed"]), device=device,
        )
        checkpoint = checkpoint_dir / f"{variant}.pt"
        torch.save({
            "experiment": "EXP-006", "stage": 2, "protocol_revision": "v2.7",
            "split": "train", "variant": variant, "scalar_dim": scalar_dim,
            "scalar_indices": scalar_indices,
            "router": final_model.state_dict(), "scalar_mean": final_mean.cpu(),
            "scalar_std": final_std.cpu(), "steps": int(config["stage2"]["steps"]),
            "query_or_future_input": False,
        }, checkpoint)
        results[variant] = {
            "scalar_dim": scalar_dim, "scalar_indices": scalar_indices,
            "metrics": metrics, "folds": fold_records,
            "final_logs": final_logs, "checkpoint": str(checkpoint),
            "rows": [{
                "fold": row["fold"], "episode": row["episode"], "candidate": row["candidate"],
                "future_utility": row["future_utility"],
                "utility_hat": float(utility_hat[index]), "risk_probability": float(risk_hat[index]),
            } for index, row in enumerate(rows)],
        }
        print(json.dumps({"variant": variant, "metrics": metrics}), flush=True)
    result = {
        "experiment": "EXP-006", "stage": "stage2_neural_router_crossfit",
        "split": "train", "protocol_revision": "v2.7",
        "source_features": config["stage2"]["source_features"],
        "validation_accessed": False, "query_or_future_router_input": False,
        "steps": int(config["stage2"]["steps"]),
        "utility_threshold": float(config["stage2"]["utility_threshold"]),
        "risk_threshold": float(config["stage2"]["risk_threshold"]),
        "variants": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"out": str(output)}))


if __name__ == "__main__":
    main()
