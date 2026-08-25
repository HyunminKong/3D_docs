# Paper Scope — Revisit3D

Last updated: 2026-08-25

## Current decision status

The frozen EXP-015 + EXP-019 candidate is **not paper-ready**. EXP-020 retained
significant proxy utility and aligned-AbsRel evidence, but failed the registered
broad-geometry and harm gates. The title and thesis below remain the research
target, not an accepted final model. Any replacement must be selected on new
development data and tested on a newly locked component-disjoint benchmark.

## Working title

**Revisit3D: Utility-Addressed Test-Time Plasticity for Streaming 3D Reconstruction**

## One-sentence thesis

Streaming 3D systems should retrieve **how to adapt** under a recurring geometric context, rather than only retaining what the scene looked like; a transported local fast code addressed by causal future utility improves later geometry under a fixed memory budget.

## Primary venue orientation

The first target is a CVPR-style paper: one compact method, standard geometry metrics, causal streaming evaluation, strong closest-method comparisons, and transparent efficiency. An ICLR submission becomes credible only if the same learning principle transfers across another backbone/task or receives substantially stronger theory. The current project must not grow into a joint static/dynamic/pose/4D system for this paper.

## Minimal method

The paper contains three ideas, but only two learned components:

1. **Local plasticity atom.** A frozen geometry backbone exposes dense features. One self-supervised gradient step updates an 8-D per-token code; backbone weights never change online.
2. **Utility retrieval.** One factorized linear pair score retrieves five predicted-utility records. A parameter-free current-geometry agreement reranks them, with coarse top-1 fallback; there is no learned fine head.
3. **Bounded causal store.** A fixed-capacity reservoir is an implementation constraint, not a novel learned consolidation module.

The primary path excludes DINO place retrieval, predicted Sim(3) transport, neural risk classification, learned eviction, pose adaptation, a second TTT step, and dynamic 4D state.

## Minimal online objective

EXP-011 selected one deployed TTT signal:

\[
\mathcal L_{\mathrm{TTT}}=\mathcal L_{\mathrm{track3D}}.
\]

Only the local code `z` is differentiated. Smoothness and code regularization were numerically immaterial in the registered train audit and are removed.

The EXP-015 core atom uses normalized future current loss, absolute best-reuse loss, and their unweighted softplus ranking. These are three readouts of the same 3D-track signal, with no auxiliary key, neutralization, centering, smoothness, or code-norm loss.

## Main hyperparameters

Only four quantities are treated as visible method hyperparameters in the frozen paper candidate:

- one-step TTT step size `eta`;
- reuse residual strength `alpha`;
- coarse candidate count `K`;
- bank capacity `C`.

Both utility and current-agreement decisions use semantic zero thresholds. PCA dimension and Ridge regularization are fixed implementation choices. The paper reports sensitivity for `eta`, `alpha`, and `C`, with a compact K ablation.


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

where `F_t` is evaluated on disjoint future/query observations. At runtime, neither `F_t` nor query frames are available. The unified address estimates utility from current/source observables.

## Theoretical rationale and claim boundary

If future loss `F` is L-smooth, then for a transported residual `delta`:

\[
F(z)-F(z+\delta)
\ge
-\langle\nabla F(z),\delta\rangle-\frac{L}{2}\lVert\delta\rVert^2.
\]

Thus useful reuse requires directional agreement with future improvement, while the fixed residual and code clamp bound the second-order damage term. This motivates learning future utility rather than using appearance similarity or raw gradient cosine. It is a rationale, not a worst-case safety guarantee.

The linear score is trained as a conditional-utility regressor and uses the semantic decision `r(o)>0`. This is Bayes-consistent under squared loss when the regressor estimates `E[U|o]`, but finite-sample Ridge has no uniform calibration guarantee. Component-disjoint utility, harmful-rate, and absolute-geometry tests are therefore mandatory; the zero rule is not presented as worst-case safety.

The linear pair score over `[c_t,c_i,c_t-c_i,c_t*c_i]` factorizes exactly into maximum inner product search. Reservoir sampling supplies fixed memory and unbiased historical inclusion, but no superiority over FIFO is claimed.

## Required paper evidence

1. Absolute depth/point metrics, not only the future-loss utility proxy.
2. Component-disjoint and source-entity-safe generalization with paired confidence intervals.
3. `base → current TTT → random memory → appearance address → utility address` ablation.
4. Comparison with frozen VGGT/FastVGGT, current-only TTT, CUT3R/TTT3R-style streaming baselines where protocols permit, and bounded/unbounded memory controls.
5. Runtime, peak GPU memory, per-record bank bytes, and sequence-length scaling.
6. At least one independent dataset or backbone before a strong general claim.
7. A minimal-loss refit showing that feasibility-stage regularizers are not the source of the result.

## Stop conditions

- If utility improvement does not reduce scale-invariant or scale-aligned geometry error, the current method is not paper-ready and the proxy claim must be narrowed.
- If utility address does not beat a matched random address on absolute metrics, similarity-independent utility retrieval is not established.
- If the simplified meta-objective collapses, retain only independently justified regularizers and report them transparently.
- No result from the closed EXP-009 test may be used to tune the method.

EXP-020 triggered the first two stop conditions, and EXP-022 showed significant
anti-alignment between proxy gain and SILog/EPE gain. Consequently, a fully
self-supervised proxy-only endpoint is rejected for the primary CVPR path. The
only active alternative is one explicitly metric-aligned offline meta label
while retaining the single self-supervised online TTT loss. Its oracle utility
must pass EXP-023 before any new model training.

EXP-023 passed the oracle gate across all three primary metrics. One
component-OOF atom fit with that single offline loss is now authorized; address
and terminal evaluation remain locked until its absolute-geometry gates pass.

That EXP-024 fit failed aligned AbsRel while significantly improving SILog/EPE.
One terminal equal-weight log-plus-relative geometry objective is authorized as
EXP-025. It keeps one online loss and adds no module or inference hyperparameter;
failure stops method development for this paper.
