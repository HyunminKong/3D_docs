# EXP-024 — Metric-Aligned Plasticity Atom

Status: Registered before execution

## Question

Can the unchanged one-step self-supervised online update learn an 8-D local
code/readout whose current and reusable effects improve absolute geometry when
the atom is meta-trained with one correctly aligned offline loss?

## Frozen method choices

- frozen VGGT/FastVGGT features and frozen PCA visual key;
- 8-D local code and the existing 157,121-parameter head;
- exactly one online track3D step at `eta=0.0125`;
- visual transport and residual `alpha=0.10`;
- the same five train-only candidate construction;
- 1000 updates, AdamW settings, and five physical-component folds from EXP-015.

## Only training change

Replace the three EXP-015 proxy readouts with one sparse metric loss. The outer
objective is the equal mean of its current and minimum-candidate evaluations:

\[
L_{outer}=\tfrac12\left(L_{metric}(z_t)+
\min_i L_{metric}(z_t+0.1T(z_i))\right).
\]

There is no proxy term, ranking term, auxiliary loss, metric weight, risk head,
or new inference parameter. LiDAR is used only on disjoint query frames during
offline meta-training/evaluation.

## Registered OOF gate

Across at least 200 episodes/20 components:

1. current TTT must improve mean SILog, aligned AbsRel, and 3D EPE over frozen
   foundation, with at least one positive component interval;
2. metric-oracle reuse must improve all three means over current, with at least
   one positive component interval;
3. oracle metric risk must beat uniform candidate expectation with a positive
   component interval.

Failure ends this from-scratch one-loss atom. Passing authorizes only a final
train refit checkpoint; utility-address fitting remains a separate experiment.
EXP-020 and the locked EXP-021 terminal set remain inaccessible.

## Files

- Config: `configs/EXP-024_metric_aligned_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp024_metric_aligned_atom.py`
- Checkpoint: `revisit3d/checkpoints/exp024_metric_aligned_atom_v10.pt`
- Result: `revisit3d/results/EXP-024/stage0_metric_aligned_atom_train_v10.json`
