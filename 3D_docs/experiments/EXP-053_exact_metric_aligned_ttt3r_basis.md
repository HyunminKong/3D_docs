# EXP-053 — Exact Metric-Aligned TTT3R Basis

Status: Completed; registered gate failed and no checkpoint was created
Purpose: Test whether one 6,144-parameter basis can align a single deployed TTT
step with absolute 3D geometry on a disjoint train scene

## Protocol

Fit on 48 deterministic anchors from `pumpkin`, `heads`, and `chess` and audit
once on 16 anchors from `stairs`. Every anchor is a four-frame reset TTT3R
segment. All scenes belong to the frozen EXP-051 train role; validation and
terminal remain closed. Missing RGB-registered depth may be generated only for
these selected train frames with the official fixed 7Scenes calibration and
must be content-inventoried.

Only the shared 8-to-768 projection (`6,144` weights) is updated. For each fit
anchor, differentiate the single median-scale-aligned relative 3D point loss
through the same one online consistency step used at deployment. Use one AdamW
pass, learning rate `1e-4`, zero weight decay, 48 updates, and no scheduler,
auxiliary loss, coefficient, second step, or other module.

Before and after fitting, evaluate the identical 16 audit anchors. Compare
against the exact initial basis, not merely the no-TTT output. The checkpoint is
written only if all registered gates pass.

## Registered gates

- exact 48-fit/16-audit coverage, finite values, changed basis, and zero-code
  parity within `1e-5`;
- final online consistency gain is positive;
- final absolute 3D gain over zero code has a positive anchor-bootstrap 95% CI;
- paired final-minus-initial absolute 3D gain has a positive 95% CI;
- final metric harm is at most 25% and no worse than the initial basis.

Passing authorizes freezing one refit protocol and then one-shot access to the
single EXP-051 validation scene. Failure stops this optimizer/objective
realization without learning-rate or loss tuning.

## Artifacts

- Config: `configs/EXP-053_exact_metric_aligned_ttt3r_basis_v10.yaml`
- Depth preparation: `revisit3d/scripts/prepare_exp053_selected_train_depth.py`
- Runner: `revisit3d/scripts/fit_exp053_exact_metric_aligned_ttt3r_basis.py`
- Result: `revisit3d/results/EXP-053/exact_metric_aligned_ttt3r_basis_v10.json`

## Result

The 48-step fit and both 16-anchor audits completed with finite gradients,
exact zero-code parity, and a basis L2 change of `0.0612`. Online consistency
gain remains positive (`6.22e-5`). The learned basis moves the mean absolute-3D
gain from `-1.42e-6` to `+0.55e-6` and improves the paired mean by `1.97e-6`.
Harm falls from 68.75% to 56.25%.

These directional improvements are not robust enough to pass. The final gain
CI is `[-2.10e-6, 3.43e-6]`, the paired final-minus-initial CI is
`[-0.67e-6, 4.69e-6]`, and final harm remains above the registered 25% maximum.
The two audit sequences also disagree: `stairs/seq-01` has positive mean gain,
whereas `stairs/seq-02` remains negative with 75% harm. Nine of 16 anchors
improve relative to the initial basis, but only seven of 16 have positive final
absolute gain.

No checkpoint was written and validation remains unopened. The result shows
that the exact meta-gradient rotates the basis in the desired average direction
but does not learn a reliably metric-aligned update from this single-pass,
single-shared-basis realization. Per D135, its learning rate, budget, seed, and
loss may not be repaired post hoc.
