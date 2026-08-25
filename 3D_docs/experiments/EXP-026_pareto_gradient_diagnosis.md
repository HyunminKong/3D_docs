# EXP-026 — Pareto Gradient Diagnosis

Status: Registered; not yet executed

## Question

Did EXP-024/025 fail because the two offline geometry endpoints produce
materially conflicting head gradients, and is there a non-degenerate update
direction that is first-order descent for both without a loss weight?

## Novelty boundary

Gradient surgery, Pareto multi-task optimization, and gradient-consensus TTA
already exist. PCGrad, CAGrad, Aligned-MTL, GraTa, ConFIG, and CoCo-MT-TTA
preclude a generic claim that aligning gradients during adaptation is new.
EXP-026 therefore treats common descent as a diagnostic and possible optimizer
primitive, not as the paper's standalone novelty.

The branch remains geometry-specific: two offline endpoint gradients train one
spatial local-code plasticity map, while deployment still performs exactly one
self-supervised `track3D` local-code step. Sparse LiDAR, future frames, and both
endpoint gradients are absent online.

## No-fit protocol

Use all 225 existing train episodes and 25 immutable physical-overlap
components. The official-test EXP-021 benchmark remains unopened. At each of
three frozen parameter anchors—fresh PCA-initialized head, EXP-006 head, and
EXP-015 head—construct the unchanged five-candidate rollout and compute:

\[
J_{log}=\tfrac12(L^{cur}_{log}+\min_i L^i_{log}),\qquad
J_{rel}=\tfrac12(L^{cur}_{rel}+\min_i L^i_{rel}).
\]

The minima are objective-specific. Both losses use detached per-view median
scale alignment and sparse query LiDAR only as offline labels. No optimizer
step is performed.

For gradients `g_log` and `g_rel`, record gradient norms, cosine, the behavior
of the raw equal average, exact two-objective raw MGDA, and the parameter-free
unit-normalized bisector

\[
d=\frac{g_{log}}{\lVert g_{log}\rVert}+
  \frac{g_{rel}}{\lVert g_{rel}\rVert}.
\]

Except at antiparallel degeneracy, `d` has positive inner product with both
unit gradients. EXP-026 measures whether that useful condition actually holds
at the local plasticity head rather than assuming it.

## Registered gate

The next fit is authorized only if all hold:

1. at least 200 targets and 20 components are valid at every anchor;
2. at least one anchor has component-balanced negative-cosine conflict rate
   at least 10%;
3. at least one anchor has raw equal-average endpoint-sacrifice rate at least
   10%;
4. normalized bisector is strict common descent on at least 95% at every
   anchor;
5. median bisector norm ratio is at least 0.05 at every anchor.

Failure means the observed terminal trade-off is not adequately explained by
the proposed local gradient mechanism, so EXP-027 must not be fit.

## Files

- Config: `configs/EXP-026_pareto_gradient_diagnosis_v10.yaml`
- Runner: `revisit3d/scripts/diagnose_exp026_pareto_gradients.py`
- Result: `revisit3d/results/EXP-026/stage0_pareto_gradient_diagnosis_v10.json`
