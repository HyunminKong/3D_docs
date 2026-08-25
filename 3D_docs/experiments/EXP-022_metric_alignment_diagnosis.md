# EXP-022 — Metric-Alignment Diagnosis

Status: Completed; metric/proxy misalignment confirmed

## Question

Did the terminal proxy-oriented atom lose metric geometry already on training
contexts, or did the final refit fail only when transferred to unseen validation
components? Is any loss caused mainly by the learned zero-code readout or by the
single online TTT step?

## Protocol

Use the existing 218 unique train targets/25 components and sparse-LiDAR query
evaluation. Do not train or select a model. Compare:

1. frozen foundation depth;
2. the metric-healthy EXP-011 reference current TTT result;
3. the EXP-015 head at zero local code;
4. the EXP-015 head after the frozen one-step `track3D` update.

For the final head, decompose `base → zero-head → current TTT`, and correlate
the disjoint-query track3D improvement with SILog, aligned AbsRel, and 3D EPE
improvement. Component bootstrap intervals quantify every comparison. Query
LiDAR and future loss remain evaluation labels and never enter online TTT.

This is a diagnosis, not another atom variant. Its registered gate checks only
coverage and exact reproduction of the existing foundation baseline.

## Files

- Config: `configs/EXP-022_metric_alignment_diagnosis_v10.yaml`
- Evaluator: `revisit3d/scripts/diagnose_exp022_metric_alignment.py`
- Result: `revisit3d/results/EXP-022/stage0_metric_alignment_diagnosis_train_v10.json`

## Result

Coverage and exact baseline-reproduction gates passed on all 218 targets/25
components. The EXP-015 zero-code output was numerically identical to frozen
foundation depth, so the static readout introduced no geometry change.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation / final zero-code | 50.3077 | 0.75538 | 6.12068 |
| EXP-011 reference current | 50.2143 | 0.74875 | 6.10439 |
| EXP-015 final current | 50.4629 | 0.74170 | 6.15514 |

The final one-step update improved aligned AbsRel strongly, but worsened the
target-average SILog and 3D EPE. Its self-supervised future improvement was
significantly anti-correlated with actual improvement for SILog
(`rho=-0.285`, `p=2.0e-5`) and 3D EPE (`rho=-0.276`, `p=3.6e-5`). The AbsRel
association was weakly positive (`rho=0.125`, `p=0.064`).

## Interpretation

The failure is not a static decoder bias and cannot be fixed by another address
or memory policy. The learned code-to-depth plasticity direction and its
track3D future target preferentially optimize scale-aligned relative depth while
being anti-aligned with scale-invariant and point geometry on individual
targets. EXP-020 is therefore a reproduced objective-alignment failure rather
than unexplained holdout noise.

## Conclusion

The self-supervised track3D future loss is retained as the single online TTT
signal, but is rejected as the paper's sole offline utility/meta target. Before
any new fit, EXP-023 must test whether one sparse scale-aligned log-depth target
can identify candidate reuse that improves all primary geometry metrics.
