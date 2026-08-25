# EXP-023 — Metric-Utility Oracle

Status: Completed; all gates passed

## Question

Can one scalar sparse metric-geometry label select frozen local corrections that
improve SILog, aligned AbsRel, and 3D EPE together, and does it select better
geometry than the rejected future-track3D proxy?

## Metric label

For each disjoint query view, median-align predicted depth to sparse LiDAR and
compute mean absolute log-depth residual over valid cells. Average the four view
losses:

\[
L_{\mathrm{metric}} = \frac{1}{V}\sum_v
\operatorname{mean}_{p\in\Omega_v}
|\log(s_v d_p)-\log d_p^*|,
\quad
s_v=\operatorname{median}_{p\in\Omega_v}(d_p^*/d_p).
\]

This is one scale-aligned loss with no metric weights. It is label-only here;
LiDAR never enters current TTT or candidate construction.

## Protocol

Freeze the EXP-015 head, one online track3D step, visual transport, residual
0.10, and the existing five candidates (`matched`, `distant`, three foreign).
On all 225 train episodes/25 components compare:

- current-only TTT;
- candidate with minimum sparse metric loss (metric oracle);
- candidate with minimum future track3D loss (proxy oracle);
- uniform candidate expectation.

The gate requires coverage, improvement of all three primary metric means over
current, at least one positive component interval, no worse primary means than
the proxy oracle, and a positive metric-risk interval over the proxy oracle.
Failure ends this scalar label; it may not be repaired by metric weighting.
Passing authorizes a component-OOF atom feasibility fit using this same single
loss, but does not authorize address fitting or terminal-test access.

## Files

- Config: `configs/EXP-023_metric_utility_oracle_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp023_metric_utility_oracle.py`
- Result: `revisit3d/results/EXP-023/stage0_metric_utility_oracle_train_v10.json`

## Result

All gates passed on 225 episodes/25 components.

| Policy | metric risk | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|---:|
| current TTT | 0.379750 | 53.3668 | 0.79276 | 6.51368 |
| metric oracle | 0.379374 | 53.3186 | 0.79074 | 6.50106 |
| proxy oracle | 0.379896 | 53.3700 | 0.79122 | 6.51430 |
| uniform candidate expectation | 0.379779 | 53.3617 | 0.79134 | 6.51212 |

Metric-oracle improvements over current had positive component intervals for
all three primary metrics: SILog `[0.0168,0.0928]`, AbsRel
`[0.00108,0.00317]`, and 3D EPE `[0.00286,0.02572]`. It also beat the proxy
oracle significantly on metric risk, SILog, and EPE. Candidate metric utility
was nevertheless harmful in 68% of the 1,125 frozen candidate applications.

## Interpretation

The single label is aligned with all three paper metrics and separates a useful
candidate from the rejected track3D proxy. The high candidate harm confirms
that retrieval/routing is necessary; it does not justify unconditional reuse.
The effect is an oracle upper bound over frozen candidates and does not yet show
that a head or address can learn the label out of component.

## Conclusion

The scalar metric label is admitted for one component-OOF atom feasibility fit.
The fit must replace, not augment, the EXP-015 proxy meta-objective. Online TTT
remains one self-supervised track3D loss and no inference module is added.
