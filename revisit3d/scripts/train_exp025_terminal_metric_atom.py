#!/usr/bin/env python3
"""Terminal equal log-plus-relative metric atom, reusing EXP-024 protocol code."""

from __future__ import annotations

import numpy as np
import torch

from revisit3d.scripts import train_exp024_metric_aligned_atom as protocol


def _metric_loss(prediction, target, valid, config):
    rows = []
    minimum_cells = int(config["lidar"]["minimum_cells_per_view"])
    epsilon = float(config["meta_objective"]["minimum_depth"])
    log_weight = float(config["meta_objective"]["log_relative_weight"])
    relative_weight = float(config["meta_objective"]["absolute_relative_weight"])
    if log_weight != 0.5 or relative_weight != 0.5:
        raise RuntimeError("EXP-025 metric weights must remain fixed at 0.5/0.5")
    for view in range(prediction.shape[0]):
        mask = valid[view]
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].clamp_min(epsilon)
        gt = target[view][mask].clamp_min(epsilon)
        scale = torch.median(gt / pred).detach()
        aligned = (pred * scale).clamp_min(epsilon)
        log_residual = (torch.log(aligned) - torch.log(gt)).abs().mean()
        relative_residual = ((aligned - gt).abs() / gt).mean()
        rows.append(0.5 * (log_residual + relative_residual))
    if not rows:
        raise RuntimeError("query LiDAR has no valid view")
    return torch.stack(rows).mean()


def _risk(prediction, target, valid, config):
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(valid):
        valid = valid.detach().cpu().numpy()
    rows = []
    minimum_cells = int(config["lidar"]["minimum_cells_per_view"])
    epsilon = float(config["meta_objective"]["minimum_depth"])
    for view in range(prediction.shape[0]):
        mask = valid[view] & np.isfinite(prediction[view]) & (prediction[view] > epsilon)
        if int(mask.sum()) < minimum_cells:
            continue
        pred = prediction[view][mask].astype(np.float64)
        gt = target[view][mask].astype(np.float64)
        scale = float(np.median(gt / pred))
        aligned = np.clip(pred * scale, epsilon, None)
        log_residual = float(np.mean(np.abs(np.log(aligned) - np.log(gt))))
        relative_residual = float(np.mean(np.abs(aligned - gt) / gt))
        rows.append(0.5 * (log_residual + relative_residual))
    return float(np.mean(rows)) if rows else None


if __name__ == "__main__":
    protocol._metric_loss = _metric_loss
    protocol._risk = _risk
    protocol.main()
