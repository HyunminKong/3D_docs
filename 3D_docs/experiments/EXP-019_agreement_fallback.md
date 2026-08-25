# EXP-019 — Geometry-Agreement Reranking with Fallback

Status: Completed; all gates passed and artifact frozen

## Question

Can current-geometry agreement improve coarse utility retrieval without EXP-018's excessive abstention?

## Method

Retrieve the same utility top-5. If any candidate has both positive predicted utility and positive current-geometry agreement, select the largest agreement; otherwise fall back to the coarse top-1 decision. This is a deterministic completion of EXP-018 with no learned model, parameter, or threshold.

The unchanged utility/acceptance/harm gates apply. In addition, component-bootstrap lower bounds must be positive against both matched-acceptance random selection and coarse top-1. Failure formally establishes that current-loss heuristics cannot replace a learned post-transport utility model.

## Result

The method achieved `+0.8351%` utility, 18.45% harm, and 94.72% acceptance. It rerouted 42 targets and used coarse fallback on 161. Full minus matched-acceptance random was `+0.00280`, 95% CI `[+0.00041,+0.00512]`; full minus coarse top-1 was `+0.00064`, CI `[+0.00010,+0.00147]`. Every gate passed and an exact-MIPS all-train artifact was frozen.

## Files

- Config: `configs/EXP-019_agreement_fallback_v10.yaml`
- Evaluator: `revisit3d/scripts/fit_exp019_agreement_fallback.py`
- Result: `revisit3d/results/EXP-019/stage0_agreement_fallback_train_v10.json`
