# EXP-023 — Metric-Utility Oracle

Status: Registered before execution

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
