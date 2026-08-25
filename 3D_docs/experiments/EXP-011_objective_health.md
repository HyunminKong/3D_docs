# EXP-011 — Minimal Online-Objective Health

Status: Registered before execution

## Question

Can one self-supervised frozen-track loss make a single local-code TTT step improve scale-invariant depth, aligned relative depth, and 3D endpoint accuracy together?

## Motivation

EXP-010 showed that the existing track-3D objective improves aligned AbsRel but worsens SILog/RMSE, and memory amplifies the mismatch. Adding more memory modules is therefore prohibited until the online objective itself is metric-healthy.

## Train-only protocol

Use the 225 EXP-009 train pilot directions, deduplicated to their unique A′ targets and grouped by 25 physical overlap components. Sparse LiDAR is evaluation/model-selection supervision only and never enters the online update.

Compare:

- registered full objective: absolute 3D track residual + smoothness + code regularization;
- absolute 3D track residual alone;
- symmetric frozen-track reprojection residual alone.

Each objective uses exactly one TTT step. The fixed step-size grid is `{0.0125, 0.025, 0.05}`. No loss weights are introduced.

## Selection and gate

For every variant, compare base versus current-only TTT on SILog, median-aligned AbsRel, and median-aligned same-ray 3D EPE. A healthy variant must:

1. cover at least 180 targets and 20 components;
2. improve the mean of all three primary metrics;
3. have a strictly positive paired component-bootstrap lower bound for at least one metric.

Among healthy variants, select the one maximizing its worst relative improvement across the three metrics. Exact ties prefer a single-loss objective and then the smaller step size. If none passes, the current depth-only plasticity formulation is stopped for the paper.

The selected objective, if any, must be frozen and evaluated once on validation before atom/router retraining. EXP-009 test is not accessed.

## Files

- Config: `configs/EXP-011_objective_health_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp011_objective_health.py`
- Result: `revisit3d/results/EXP-011/stage0_objective_health_train_v10.json`
