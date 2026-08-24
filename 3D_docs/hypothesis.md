# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Supported for static revisits.** Global and slot states failed context selectivity, whereas visual local-code transport produced positive utility throughout the expanded train, unseen validation, and final unseen test. On the locked test, the selected bounded system reached +2.088% routed utility over current-only TTT and the top-K oracle reached +3.356%.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected.** Predicted geometry and geometry+appearance transport did not improve over visual transport, reduced coverage, and increased harm. Oracle-coordinate EXP-005 remains an upper bound, not deployable evidence.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is necessary primary evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** No-alignment observable routing generalized, while geometry-only routing was weaker and less safe. Alignment statistics remain diagnostic ablations.

## H3 — Online utility observability

Current/source descriptors, current-only loss changes, update statistics, and transport evidence predict the future utility of a past local correction without accessing future frames online.

Status: **Supported.** The source-entity-safe leave-one-location-out transport-descriptor address had positive utility association in every held location and beat matched random retrieval. The frozen address/router then transferred to validation and a terminal 22-component test. Query/future quantities were used only for offline labels and evaluation.

## H4-U — Learnable utility addressing and routing

A trainable candidate/current utility model can retrieve and apply more useful adaptation than identity, appearance/place similarity, current-loss heuristics, and matched random addressing.

Status: **Supported for ranking and bounded application.** The exact 64-D utility-MIPS address plus fixed utility router achieved +2.088% on the locked test versus +1.602% for same-bank random addressing. The paired 22-component difference was +0.475 percentage points with 95% CI [+0.086, +0.924]. The router rejected some candidates but still produced 3.85% deadband harm, so perfect safety is not supported.

## H4-R — Learnable negative-transfer risk

A separate risk classifier can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the tested classifier.** Neural risk AUROC was nontrivial, but hard routing did not improve selected harm over the compact utility router. Safety in the final model comes from utility selection, current-only fallback, and a fixed 0.10 residual—not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

A causal, capacity-bounded memory can retain most of an unbounded utility-addressed bank's revisit benefit.

Status: **Supported at the EXP-009 static-benchmark scale; partially supported as a general continual-learning claim.** Reservoir-64 retained 100.98% of unbounded routed utility on the locked test and beat matched random addressing with a positive component interval. However, reservoir exceeded FIFO by only +0.020 percentage points and its CI crossed zero. Capacity 64 was selected on validation and is not universal; learned history eviction and generic DINO place consolidation were rejected.

## H6 — Extension to dynamic 4D

The same utility-routed local-memory principle can attach to tracked dynamic points or motion-conditioned regions to improve reappearance and occlusion recovery.

Status: **Open; outside the completed static-revisit milestone.**

## Rejected or narrowed hypotheses

- A small global/slot fast-weight vector is sufficiently context-selective for retrieval.
- More optimization alone fixes update-direction collapse.
- Cosine similarity of raw gradients is a reliable proxy for causal future utility.
- The paired physical revisit is always the uniquely correct or most useful memory.
- A parameter-free current-loss threshold is a sufficient negative-transfer safeguard.
- Predicted 3D correspondence should be the primary carrier of the fast update.
- Geometry-alignment failure should hard-reject a candidate that still has valid visual transport.
- A second current TTT step is an equivalent replacement for memory reuse.
- Generic DINOv2 place compatibility is a causal adaptation-utility address.
- Past utility history is a reliable learned eviction priority at capacity 8.
- Capacity 8 is a general sufficient bound.
- Reservoir sampling is demonstrably superior to FIFO at capacity 64; the final test does not support this stronger claim.
