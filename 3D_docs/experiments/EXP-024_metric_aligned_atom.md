# EXP-024 — Metric-Aligned Plasticity Atom

Status: Completed; registered gate failed

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

## Result

The five-fold OOF run covered all 225 episodes/25 components. No final
checkpoint was produced because one registered check failed.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation | 53.4149 | 0.81202 | 6.53079 |
| current TTT | 52.8313 | 0.81893 | 6.34602 |
| metric oracle | 52.6243 | 0.81699 | 6.29415 |
| uniform candidate expectation | 52.7479 | 0.81826 | 6.32616 |

Current TTT significantly improved SILog and 3D EPE, but worsened aligned
AbsRel by 0.00691. Oracle reuse improved all three metrics over current with
strictly positive component intervals, and oracle metric risk beat uniform
candidates with interval `[0.00099,0.00170]`. Mean candidate metric utility was
positive and candidate harm fell from EXP-023's 68% to 30.93%.

## Interpretation

The single log residual learns broadly useful current/reuse directions but does
not control the one metric it omits explicitly: aligned relative depth. The
memory mechanism and selection headroom are healthy; the current-quality gate
fails only on AbsRel. This is not permission to restore proxy terms or add a
router.

## Conclusion

The from-scratch one-loss atom is rejected and no checkpoint is accepted.
EXP-025 is a terminal objective test: add only the aligned relative-depth
residual to the log residual with fixed equal averaging. No other objective
variant is permitted for this paper if EXP-025 fails.
