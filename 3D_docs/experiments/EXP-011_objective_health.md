# EXP-011 — Minimal Online-Objective Health

Status: Completed; train selection and one-shot validation gates passed

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

## Stage 0 result — train

All 218 unique targets across 25 physical components had valid sparse-LiDAR evaluation. The registered gate passed and selected **one absolute 3D frozen-track consistency loss, one TTT step, and `eta=0.0125`**.

| Variant | SILog | AbsRel | 3D EPE (m) | Gate |
|---|---:|---:|---:|---|
| no TTT | 50.3077 | 0.75538 | 6.12068 | — |
| registered composite, 0.0125 | 50.2144 | 0.74873 | 6.10440 | pass |
| **3D track only, 0.0125** | **50.2143** | **0.74875** | **6.10439** | **selected** |
| reprojection only, 0.0125 | 50.1212 | 0.75762 | 6.07779 | fail: AbsRel |

The selected objective improved target means by 0.186% SILog, 0.878% AbsRel, and 0.266% 3D EPE. Its component-bootstrap AbsRel improvement was `+0.00569`, 95% CI `[+0.00180, +0.00961]`; SILog and EPE intervals crossed zero. Smoothness and code regularization changed the result negligibly, so they are removed from the paper online objective. Larger `eta=0.05` improved AbsRel more but worsened SILog and EPE, confirming the EXP-010 overshoot diagnosis.

## Locked Stage 1 — one-shot validation

The selected Stage-0 choice is immutable. Stage 1 uses the existing 117-direction validation pilot, deduplicated to 103 unique A-prime targets over 17 components. It compares only no-TTT and current-only one-step TTT with the frozen 3D-track-only objective at `eta=0.0125`. Sparse LiDAR remains evaluation-only.

The Stage-1 gate requires at least 85 valid targets and 15 components, improvement in the mean of all three primary metrics, and a strictly positive component-bootstrap lower bound for at least one metric. Failure stops reuse-model refitting and sends the paper to a new-data/objective decision. The terminal EXP-009 test remains closed.

## Stage 1 result — one-shot validation

All 103 unique targets across 17 components had valid LiDAR coverage. The frozen objective improved all registered target means:

| Metric | No TTT | One-step TTT | Relative improvement | Component-bootstrap improvement, 95% CI |
|---|---:|---:|---:|---:|
| SILog | 47.9994 | 47.7721 | +0.474% | +0.1033 `[-0.0066, +0.2125]` |
| AbsRel | 0.74013 | 0.73304 | +0.959% | +0.00353 `[-0.00097, +0.00728]` |
| 3D EPE (m) | 5.22834 | 5.17647 | +0.992% | +0.03291 `[+0.00120, +0.06480]` |

The registered gate passed. This supports refitting the paper-minimal atom and utility retrieval with the objective fixed. It does not authorize another EXP-009 test evaluation.

## Files

- Config: `configs/EXP-011_objective_health_v10.yaml`
- Validation config: `configs/EXP-011_locked_validation_v11.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp011_objective_health.py`
- Validation evaluator: `revisit3d/scripts/evaluate_exp011_locked_validation.py`
- Result: `revisit3d/results/EXP-011/stage0_objective_health_train_v10.json`
- Validation result: `revisit3d/results/EXP-011/stage1_locked_objective_validation_v11.json`
