# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Partially supported.** Global and slot states failed context selectivity, whereas visual local-code transport repeatedly produced positive self-supervised utility. EXP-011 showed that a single smaller 3D-track step can improve SILog, aligned AbsRel, and 3D EPE. The auxiliary-free EXP-015 refit produced OOF reuse headroom, but EXP-020 showed that this refit no longer preserved broad metric geometry. Spatial reuse remains plausible; the frozen paper-model instantiation is rejected.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected.** Predicted geometry and geometry+appearance transport did not improve over visual transport, reduced coverage, and increased harm. Oracle-coordinate EXP-005 remains an upper bound, not deployable evidence.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is necessary primary evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** No-alignment observable routing generalized, while geometry-only routing was weaker and less safe. Alignment statistics remain diagnostic ablations.

## H3 — Online utility observability

Current/source descriptors, current-only loss changes, update statistics, and transport evidence predict the future utility of a past local correction without accessing future frames online.

Status: **Partially supported for proxy utility only.** Source-entity-safe addressing predicts the self-supervised future-loss label without query leakage, including significant superiority to matched random in EXP-020. Its ability to predict broad metric reconstruction improvement is not established.

## H4-U — Learnable utility addressing and routing

A trainable candidate/current utility model can retrieve and apply more useful adaptation than identity, appearance/place similarity, current-loss heuristics, and matched random addressing.

Status: **Partially supported for proxy ranking and aligned AbsRel.** The exact utility-MIPS address and compact geometry reranking beat same-bank random on self-supervised utility. EXP-020 also significantly improved aligned AbsRel over current TTT, but did not significantly beat random on any primary LiDAR metric and retained 33% proxy harm. Broad geometry-utility routing remains unsupported.

## H4-R — Learnable negative-transfer risk

A separate risk classifier can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the tested classifier.** Neural risk AUROC was nontrivial, but hard routing did not improve selected harm over the compact utility router. Safety in the final model comes from utility selection, current-only fallback, and a fixed 0.10 residual—not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

A causal, capacity-bounded memory can retain most of an unbounded utility-addressed bank's revisit benefit.

Status: **Partially supported for proxy retention.** Reservoir-64 retained the proxy benefit under the closed EXP-009 protocol, but did not beat FIFO significantly. EXP-020 did not establish consistent metric-geometry improvement. Capacity 64 is benchmark-selected; learned history eviction and DINO place consolidation remain rejected.

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
