# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Partially supported on train OOF.** Visual local transport achieved +1.62% utility, whereas untransported local reuse was negative and a global vector gave only +0.21%. Generalization beyond 8 overlap components is open.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected in EXP-006 train OOF.** Geometry and geometry+appearance did not improve mean utility over visual transport, covered only 48% of candidates, and increased harm. Oracle-coordinate EXP-005 remains an upper-bound observation, not deployable evidence.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is useful as evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** The small 8-component increment did not reproduce after expansion: no-geometry/full utility was +2.80%/+2.82% with equal harm, while geometry-only routing was +1.84% with 6.58% harm. Alignment statistics remain diagnostic ablations only.

## H3 — Online utility observability

Current-only loss changes, update statistics, descriptors, and transport/alignment evidence predict the future utility of a past local update without accessing future frames online.

Status: **Supported on expanded train OOF.** Appearance plus current/source adaptation history selected +2.80% utility with 2.63% harm, versus +1.84%/3.95% for visual mean and +1.38%/2.63% for current-objective routing. One-shot validation remains required.

## H4-U — Learnable utility routing

A trainable candidate/current utility head can outperform current-loss, appearance-similarity, paired-identity, random-candidate, and visual-mean controls under grouped OOF evaluation.

Status: **Supported on expanded train OOF.** The locked regularized utility router reached +2.80% selected utility and 0.51% regret. Its advantage over visual mean was +0.98 percentage points with component-bootstrap 95% CI [+0.63, +1.45]. Official validation remains unopened.

## H4-R — Learnable negative-transfer risk

A risk head can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the current explicit classifier.** Expansion produced 17 harmful candidates across three folds, so identifiability passed. Neural risk AUROC reached 0.69 without alignment and 0.72 with alignment, but hard routing still caused 3.95–5.26% harm and did not beat the regularized utility router. Safety in the locked model comes from bounded residual application and conservative utility selection, not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

Merging compatible local memories and preserving uncertainty/utility statistics can bound memory while retaining revisit benefit.

Status: **Open; persistent-bank experiments are gated on H4-U/H4-R.**

## H6 — Extension to dynamic 4D

The same utility-routed local-memory principle can attach to tracked dynamic points or motion-conditioned regions to improve reappearance and occlusion recovery.

Status: **Open; outside the static-revisit milestone.**

## Rejected hypotheses

- A small global/slot fast-weight vector is sufficiently context-selective for retrieval.
- More optimization alone fixes update-direction collapse.
- Cosine similarity of raw gradients is a reliable proxy for causal future utility.
- The paired physical revisit is always the uniquely correct or most useful memory.
- A parameter-free current-loss threshold is a sufficient negative-transfer safeguard.
- Predicted 3D correspondence should be the primary carrier of the fast update.
- Geometry-alignment failure should hard-reject a candidate that still has a valid visual transport.
- A second current TTT step is an equivalent replacement for memory reuse.
