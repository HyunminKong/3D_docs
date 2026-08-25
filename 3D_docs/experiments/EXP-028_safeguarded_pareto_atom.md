# EXP-028 — Safeguarded Pareto Plasticity Atom

Status: Registered; not yet executed

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
