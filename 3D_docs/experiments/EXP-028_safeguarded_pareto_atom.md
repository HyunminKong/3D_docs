# EXP-028 — Safeguarded Pareto Plasticity Atom

Status: Completed; all registered gates passed

## Question

Is EXP-027's remaining AbsRel damage caused by AdamW rotating a valid common
gradient, and can a parameter-free feasible-displacement safeguard recover a
broadly healthy atom?

## Only method change

Compute the same unit-normalized common gradient and the same AdamW proposal as
EXP-027. If the negative proposal displacement has positive inner product with
both endpoint gradients, accept it unchanged. Otherwise replace only its
direction by the normalized common gradient while preserving the proposal's
L2 norm:

\[
\Delta\theta=
\begin{cases}
\Delta\theta_{AdamW}, &
\langle-\Delta\theta_{AdamW},g_j\rangle>0\;\forall j,\\
-\lVert\Delta\theta_{AdamW}\rVert\,g/\lVert g\rVert,&\text{otherwise}.
\end{cases}
\]

This adds no coefficient, threshold, line search, module, or inference
operation. The seed, folds, architecture, candidates, optimizer settings, and
1000-step budget match EXP-027 exactly, providing a paired optimizer ablation.

## Registered gate

The EXP-027 OOF gates are unchanged, except realized common descent must now be
100% by construction. Current and log-risk oracle reuse must improve mean
SILog, aligned AbsRel, and 3D EPE; component intervals and oracle-over-uniform
risk evidence remain mandatory. Failure creates no checkpoint and ends this
specific safeguarded-optimizer candidate.

## Files

- Config: `configs/EXP-028_safeguarded_pareto_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp028_safeguarded_pareto_atom.py`
- Result: `revisit3d/results/EXP-028/stage0_safeguarded_pareto_atom_train_v10.json`
- Conditional checkpoint: `revisit3d/checkpoints/exp028_safeguarded_pareto_atom_v10.pt`

## Result

All 225 OOF targets and 25 components passed. The safeguard activated on
29.28% of OOF optimizer steps and raised realized common descent from EXP-027's
72.54% to 100%.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation | 53.4149 | 0.81202 | 6.53079 |
| current TTT | 53.2398 | 0.80832 | 6.48049 |
| log-risk oracle reuse | 53.1757 | 0.80692 | 6.46334 |

Current improvements were 0.1751 SILog, 0.00370 AbsRel, and 0.05030 m EPE.
SILog and EPE component intervals were positive: `[0.01233, 0.34673]` and
`[0.01000, 0.09276]`. Oracle reuse improved all three endpoints over current
with positive intervals and beat uniform candidate log-risk with interval
`[0.00032, 0.00052]`.

The full-train checkpoint hash is
`3ebf194f3a28876014e46d1d3bbdbcd1422cfb8ebdba48f3d16635520ca787ae`.

## Conclusion

EXP-028 is the first accepted atom whose current one-step TTT and reusable
candidate headroom are jointly broad-metric healthy. The safeguard affects
offline meta-training only. The frozen inference architecture is unchanged.
Raw candidate harm remains 39.91%, so unconditional memory reuse is prohibited
and one metric-utility address must be validated next.
