# EXP-025 — Terminal Metric Plasticity Atom

Status: Registered before execution

## Question

Does one compact two-residual offline geometry objective preserve the strong
SILog/EPE gains of EXP-024 while correcting its sole aligned-AbsRel failure?

## Objective

At each valid sparse query cell, use the fixed equal average

\[
L_{geo}=\tfrac12|\log(\hat d/d^*)|+
\tfrac12|\hat d-d^*|/d^*,
\]

after the same detached per-view median scale alignment. The outer objective is
the equal mean of current and best-candidate `L_geo`. Both 0.5 coefficients are
constants, not selected hyperparameters.

## Frozen protocol and terminal rule

Architecture, 8-D code, frozen PCA key, one online track3D step, eta 0.0125,
visual residual 0.10, five candidates, five component folds, 1000 updates, and
optimizer are identical to EXP-024. No proxy, ranking, key, neutralization,
regularization, risk, or routing term exists.

The registered OOF gates are identical to EXP-024. Current TTT and metric-oracle
reuse must each improve all three primary means, with positive component
interval evidence, and the oracle must beat uniform candidate risk. Failure
ends atom objective development for this paper. Success freezes one full-train
checkpoint and authorizes metric-utility address fitting only.

## Files

- Config: `configs/EXP-025_terminal_metric_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp025_terminal_metric_atom.py`
- Checkpoint: `revisit3d/checkpoints/exp025_terminal_metric_atom_v10.pt`
- Result: `revisit3d/results/EXP-025/stage0_terminal_metric_atom_train_v10.json`
