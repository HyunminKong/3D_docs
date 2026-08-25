# EXP-020 — Locked Paper-Model Validation

Status: Registered before execution

## Question

Does the frozen compact method improve both causal future utility and sparse-LiDAR geometry on unseen validation components?

## Frozen method

- frozen FastVGGT/VGGT features and depth;
- 8-D local code, one 3D-track TTT step, `eta=0.0125`;
- visual local-code transport with residual `alpha=0.10`;
- one 64-D utility-MIPS Ridge, top-5;
- parameter-free positive current-geometry reranking with coarse top-1 fallback;
- deterministic reservoir capacity 64 per official-location stream.

There is no fine router, risk head, learned threshold, extra online loss, or learned eviction.

## One-shot validation protocol

Replay the existing validation contexts in capture-time order, partitioned by official location. Evaluate each target before write. Query frames and LiDAR are offline labels only. Compare no-TTT, current-only TTT, coarse address, appearance, matched-acceptance random memory, and the full frozen method.

The gate requires current TTT to improve mean SILog/AbsRel/3D EPE over base; full memory not to worsen any of those means versus current or random; at least one positive component-bootstrap interval over each; proxy superiority to random with a positive interval; no more than 20% proxy harm; and at least 20% acceptance. No validation result may change the frozen model. EXP-009 test remains terminal and closed.

## Files

- Config: `configs/EXP-020_paper_model_validation_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp020_paper_model_validation.py`
- Result: `revisit3d/results/EXP-020/stage0_locked_paper_model_validation_v10.json`
