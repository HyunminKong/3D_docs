# EXP-025 — Terminal Metric Plasticity Atom

Status: Completed; terminal gate failed

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

## Result

The run covered all 225 OOF episodes/25 components. The gate failed, so no
full-train checkpoint was created.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation | 53.4149 | 0.81202 | 6.53079 |
| current TTT | 53.8756 | 0.78656 | 6.63025 |
| metric oracle | 53.7876 | 0.78160 | 6.60922 |
| uniform candidate expectation | 53.8774 | 0.78442 | 6.63013 |

Current aligned AbsRel improved by 0.02546 with component interval
`[0.01384,0.03735]`. In contrast, SILog worsened by 0.46068 with interval
`[-0.91777,-0.07194]`, and 3D EPE worsened by 0.09945 m with interval
`[-0.19515,-0.00595]`. Metric-oracle reuse still improved all three metrics
over current with positive intervals, and beat uniform candidate risk.

## Interpretation

EXP-024 and EXP-025 expose a reproducible objective conflict. Log-only
meta-training improves SILog/EPE but worsens AbsRel; adding equal relative-depth
pressure improves AbsRel but worsens SILog/EPE. Reuse headroom exists under both
heads, but a scalarized from-scratch atom does not produce a current update that
is healthy on all paper endpoints.

## Conclusion

The terminal atom gate failed and no checkpoint is accepted. Per D090/D091,
atom loss, seed, budget, initialization, or module search stops here. Continuing
requires an explicit paper-scope change, such as making constrained
multi-objective plasticity a new central contribution, or narrowing the paper
endpoint. Neither is authorized by this experiment.
