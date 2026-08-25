# EXP-019 — Geometry-Agreement Reranking with Fallback

Status: Registered before execution

## Question

Can current-geometry agreement improve coarse utility retrieval without EXP-018's excessive abstention?

## Method

Retrieve the same utility top-5. If any candidate has both positive predicted utility and positive current-geometry agreement, select the largest agreement; otherwise fall back to the coarse top-1 decision. This is a deterministic completion of EXP-018 with no learned model, parameter, or threshold.

The unchanged utility/acceptance/harm gates apply. In addition, component-bootstrap lower bounds must be positive against both matched-acceptance random selection and coarse top-1. Failure formally establishes that current-loss heuristics cannot replace a learned post-transport utility model.

## Files

- Config: `configs/EXP-019_agreement_fallback_v10.yaml`
- Evaluator: `revisit3d/scripts/fit_exp019_agreement_fallback.py`
- Result: `revisit3d/results/EXP-019/stage0_agreement_fallback_train_v10.json`
