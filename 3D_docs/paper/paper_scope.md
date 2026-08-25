# Paper Scope — Revisit3D

Last updated: 2026-08-25

## Working title

**Revisit3D: Utility-Addressed Test-Time Plasticity for Streaming 3D Reconstruction**

## One-sentence thesis

Streaming 3D systems should retrieve **how to adapt** under a recurring geometric context, rather than only retaining what the scene looked like; a transported local fast code addressed by causal future utility improves later geometry under a fixed memory budget.

## Primary venue orientation

The first target is a CVPR-style paper: one compact method, standard geometry metrics, causal streaming evaluation, strong closest-method comparisons, and transparent efficiency. An ICLR submission becomes credible only if the same learning principle transfers across another backbone/task or receives substantially stronger theory. The current project must not grow into a joint static/dynamic/pose/4D system for this paper.

## Minimal method

The paper contains three ideas, but only two learned components:

1. **Local plasticity atom.** A frozen geometry backbone exposes dense features. One self-supervised gradient step updates an 8-D per-token code; backbone weights never change online.
2. **Utility retrieval.** A cheap linear pair score retrieves K candidates, then one compact linear utility model selects a transported code or returns current-only TTT. These are presented as coarse and fine stages of one utility-retrieval module, not as unrelated heads.
3. **Bounded causal store.** A fixed-capacity reservoir is an implementation constraint, not a novel learned consolidation module.

The primary path excludes DINO place retrieval, predicted Sim(3) transport, neural risk classification, learned eviction, pose adaptation, a second TTT step, and dynamic 4D state.

## Minimal online objective

The deployed TTT objective has one main signal and two small regularizers:

\[
\mathcal L_{\mathrm{TTT}}
=
\mathcal L_{\mathrm{track3D}}
+\lambda_s\mathcal L_{\mathrm{smooth}}
+\lambda_z\lVert z\rVert_2^2.
\]

Only the local code `z` is differentiated. The main paper should report the geometry-consistency term alone, plus the full three-term objective; it should not introduce additional online losses unless the main objective fails independently.

The existing atom was meta-trained with extra protection/key regularizers developed during feasibility work. Before paper submission, a train/validation-only simplification experiment must test a two-concept meta-objective: current-quality preservation plus future reuse utility. Extra terms may remain only if an ablation demonstrates material independent value.

## Main hyperparameters

Only five quantities are treated as visible method hyperparameters:

- one-step TTT step size `eta`;
- reuse residual strength `alpha`;
- retrieved candidate count `K`;
- bank capacity `C`;
- train-calibrated utility acceptance threshold `tau`.

PCA dimension and Ridge regularization are implementation choices fixed once, not separately tuned per dataset. The paper must include sensitivity only for `eta`, `alpha`, and `C`; K receives a small `{1, 5, 10}` ablation if compute permits.

## Formal learning problem

For current context `x_t`, local adaptation is

\[
z_t=-\eta\nabla_z\mathcal L_{\mathrm{TTT}}(x_t,z)\rvert_{z=0}.
\]

A past record `i` stores a local code `z_i` and observable descriptor `c_i`. Visual transport maps it into current token coordinates:

\[
\delta_{t,i}=\alpha\,T(z_i;x_i\rightarrow x_t).
\]

Future utility is an offline meta-label only:

\[
U_{t,i}=\frac{F_t(z_t)-F_t(z_t+\delta_{t,i})}{|F_t(z_t)|+\epsilon},
\]

where `F_t` is evaluated on disjoint future/query observations. At runtime, neither `F_t` nor query frames are available. The address and router estimate utility from current/source observables.

## Theoretical rationale and claim boundary

If future loss `F` is L-smooth, then for a transported residual `delta`:

\[
F(z)-F(z+\delta)
\ge
-\langle\nabla F(z),\delta\rangle-\frac{L}{2}\lVert\delta\rVert^2.
\]

Thus useful reuse requires directional agreement with future improvement, while the fixed residual and code clamp bound the second-order damage term. This motivates learning future utility rather than using appearance similarity or raw gradient cosine. It is a rationale, not a worst-case safety guarantee.

If a calibrated router `r(o)` approximates conditional expected utility with error at most `e`, accepting only when `r(o)>tau>e` yields positive conditional expected utility at least `tau-e`. The empirical Ridge router is not claimed to satisfy a uniform error bound; calibration and harmful-rate tests are therefore mandatory.

The linear pair score over `[c_t,c_i,c_t-c_i,c_t*c_i]` factorizes exactly into maximum inner product search. Reservoir sampling supplies fixed memory and unbiased historical inclusion, but no superiority over FIFO is claimed.

## Required paper evidence

1. Absolute depth/point metrics, not only the future-loss utility proxy.
2. Component-disjoint and source-entity-safe generalization with paired confidence intervals.
3. `base → current TTT → random memory → utility address → full router` ablation.
4. Comparison with frozen VGGT/FastVGGT, current-only TTT, CUT3R/TTT3R-style streaming baselines where protocols permit, and bounded/unbounded memory controls.
5. Runtime, peak GPU memory, per-record bank bytes, and sequence-length scaling.
6. At least one independent dataset or backbone before a strong general claim.
7. A minimal-loss refit showing that feasibility-stage regularizers are not the source of the result.

## Stop conditions

- If utility improvement does not reduce scale-invariant or scale-aligned geometry error, the current method is not paper-ready and the proxy claim must be narrowed.
- If utility address does not beat a matched random address on absolute metrics, similarity-independent utility retrieval is not established.
- If the simplified meta-objective collapses, retain only independently justified regularizers and report them transparently.
- No result from the closed EXP-009 test may be used to tune the method.
