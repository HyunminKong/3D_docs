# EXP-027 — Coefficient-Free Pareto Plasticity Atom

Status: Completed; registered gate failed, no checkpoint

## Question

Can one 8-D local plasticity head learn a current update and reusable past
corrections that improve SILog, aligned AbsRel, and 3D EPE together when the
two offline endpoint gradients are balanced by direction rather than by a
scalar loss weight?

## Method change

Everything in the EXP-024/025 inference graph remains fixed. During offline
meta-training only, separately differentiate

\[
J_{log}=\tfrac12(L^{cur}_{log}+\min_iL^i_{log}),\qquad
J_{rel}=\tfrac12(L^{cur}_{rel}+\min_iL^i_{rel}),
\]

then synthesize the optimizer gradient

\[
g=\tfrac12\left(
\frac{\nabla J_{log}}{\lVert\nabla J_{log}\rVert}+
\frac{\nabla J_{rel}}{\lVert\nabla J_{rel}\rVert}
\right).
\]

This has no loss coefficient or solver hyperparameter. The same AdamW, 1000
steps, five component folds, five candidates, and all architecture settings are
frozen. Because adaptive preconditioning can rotate the realized displacement,
the experiment also measures its dot product with both original gradients.

At inference there is still only one `track3D` loss, one code step, and no
LiDAR, future frame, multi-objective gradient, or extra module.

## Registered OOF gate

All conditions are necessary:

1. at least 200 targets and 20 components;
2. current TTT improves mean SILog, aligned AbsRel, and 3D EPE over foundation;
3. at least one current improvement has a positive component-bootstrap interval;
4. log-risk oracle reuse improves all three means over current and at least one
   has a positive interval;
5. log-risk oracle beats uniform candidate expectation with a positive risk
   interval;
6. component-balanced realized AdamW displacement is common descent for at
   least 90% of training steps.

Failure rejects the fit and creates no checkpoint. Passing freezes one refit
checkpoint and authorizes metric utility-address fitting; it does not authorize
the locked EXP-021 test by itself.

## Files

- Config: `configs/EXP-027_pareto_plasticity_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp027_pareto_plasticity_atom.py`
- Result: `revisit3d/results/EXP-027/stage0_pareto_plasticity_atom_train_v10.json`
- Conditional checkpoint: `revisit3d/checkpoints/exp027_pareto_plasticity_atom_v10.pt`

## Result

The run covered all 225 OOF targets and 25 components. No checkpoint was
created.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| foundation | 53.4149 | 0.81202 | 6.53079 |
| current TTT | 53.1808 | 0.81482 | 6.44445 |
| log-risk oracle reuse | 53.0746 | 0.81317 | 6.41702 |

Current SILog improved by 0.2341 and EPE by 0.08634 m, with a positive EPE
interval `[0.03388, 0.14375]`. AbsRel still worsened by 0.00279, although its
interval `[-0.01357, 0.00584]` crossed zero. This is less than half EXP-024's
0.00691 AbsRel damage but does not pass the all-means rule.

Oracle reuse improved all three metrics over current with positive intervals,
and beat uniform candidate log-risk with interval `[0.00048, 0.00088]`.
However, although the synthesized gradient was common descent on all 5000
steps, the realized AdamW displacement was common descent on only 72.54% after
component balancing, below the registered 90% gate.

## Conclusion

Directional balancing reduces the scalarization trade-off and retains strong
memory headroom, but AdamW rotates the synthesized direction often enough to
invalidate both the optimization premise and the broad-geometry gate. The fit
is rejected. This specifically motivates testing an optimizer-level feasible
displacement, not another loss or architecture.
