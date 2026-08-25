# EXP-020 — Locked Paper-Model Validation

Status: Completed; registered paper-model gate failed

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

## Result

The frozen model was evaluated once on 103 targets from 17 unseen physical
components. The validation split is now exposed and cannot be used to tune a
replacement.

Component-balanced means were:

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| no TTT | 47.2512 | 0.67589 | 5.57367 |
| current-only TTT | 47.2797 | 0.66090 | 5.57931 |
| full memory | 47.2789 | 0.65925 | 5.57766 |
| matched-acceptance random | 47.2820 | 0.65971 | 5.57868 |
| coarse utility top-1 | 47.2762 | 0.65923 | 5.57758 |

Full memory improved aligned AbsRel over current-only TTT by 0.00165, with a
95% component-bootstrap interval `[0.00078, 0.00276]`. It did not obtain a
positive interval over random on any primary LiDAR metric. The self-supervised
utility still beat matched random by 0.00286 with interval
`[0.00120, 0.00459]`, but its harmful rate was 33.01% at 96.12% acceptance.

The registered checks failed because:

- current-only TTT did not improve all three primary means over no TTT;
- full memory did not significantly beat random on a primary LiDAR metric;
- proxy harmful rate exceeded the locked 20% maximum.

## Interpretation

The utility-addressed reuse phenomenon persists: the full method significantly
improves proxy utility over random and improves aligned AbsRel over current TTT.
However, the final EXP-015 head is not broadly metric-healthy. This differs from
the earlier EXP-011 head, whose frozen one-step objective improved all three
validation metrics. The terminal compact atom meta-objective therefore traded
metric geometry alignment for future-proxy reuse.

## Conclusion

The EXP-015 + EXP-019 combination is rejected as the paper model. The broad
streaming 3D reconstruction claim is not supported. No parameter, threshold,
loss, or routing choice may be changed using this exposed validation result. A
replacement direction requires a newly locked independent benchmark.
