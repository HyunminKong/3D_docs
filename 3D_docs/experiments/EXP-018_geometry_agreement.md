# EXP-018 — Retrieval with Current-Geometry Agreement

Status: Registered before execution

## Question

Can a parameter-free current-geometry agreement check close the negative-transfer gap left by pre-transport utility addressing?

## Method

Use EXP-016's visual-only Ridge/MIPS to retrieve top-5. For those five candidates only, apply each transported code to the current context and measure normalized reduction of the same online 3D-track loss. Select the largest agreement and reuse only when both predicted future utility and current agreement are positive. This adds no learned model, loss, learned threshold, or feature dimension; `K=5` is fixed from the prior causal benchmark.

## Registered gate

Reuse the exact source-safe folds and causal pairs. The method must exceed 0.5% utility, accept at least 20%, keep harm at or below 20%, and have positive component-bootstrap lower bounds over both matched-acceptance random selection and EXP-016 coarse top-1. Failure means a learned fine utility model is necessary and requires an explicit paper-complexity decision.

## Files

- Config: `configs/EXP-018_geometry_agreement_v10.yaml`
- Evaluator: `revisit3d/scripts/fit_exp018_geometry_agreement.py`
- Result: `revisit3d/results/EXP-018/stage0_geometry_agreement_train_v10.json`
