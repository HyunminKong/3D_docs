# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Partially supported.** Global and slot states failed context selectivity, whereas visual local-code transport produced positive self-supervised utility throughout train, validation, and final test. EXP-011 showed that a single smaller 3D-track step improves SILog, aligned AbsRel, and 3D EPE on train and one-shot validation. The auxiliary-free EXP-015 refit then achieved +1.048% component-OOF oracle reuse utility with positive selection headroom. Deployable addressing and new external absolute-geometry evidence remain required.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected.** Predicted geometry and geometry+appearance transport did not improve over visual transport, reduced coverage, and increased harm. Oracle-coordinate EXP-005 remains an upper bound, not deployable evidence.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is necessary primary evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** No-alignment observable routing generalized, while geometry-only routing was weaker and less safe. Alignment statistics remain diagnostic ablations.

## H3 — Online utility observability

Current/source descriptors, current-only loss changes, update statistics, and transport evidence predict the future utility of a past local correction without accessing future frames online.

Status: **Partially supported.** The source-entity-safe address predicted the self-supervised future-loss label and transferred to validation/test without query leakage. In EXP-010 its score was negatively associated with LiDAR metric improvements, so observability of actual reconstruction utility is not established.

## H4-U — Learnable utility addressing and routing

A trainable candidate/current utility model can retrieve and apply more useful adaptation than identity, appearance/place similarity, current-loss heuristics, and matched random addressing.

Status: **Partially supported for proxy ranking.** The exact utility-MIPS address plus fixed router beat same-bank random addressing on the registered self-supervised utility. EXP-010 found significant aligned-AbsRel improvement but worse SILog and 3D endpoint error, and router scores were negatively correlated with those metric gains. Broad geometry-utility routing is therefore open.

## H4-R — Learnable negative-transfer risk

A separate risk classifier can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the tested classifier.** Neural risk AUROC was nontrivial, but hard routing did not improve selected harm over the compact utility router. Safety in the final model comes from utility selection, current-only fallback, and a fixed 0.10 residual—not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

A causal, capacity-bounded memory can retain most of an unbounded utility-addressed bank's revisit benefit.

Status: **Partially supported for proxy retention.** Reservoir-64 retained 100.98% of unbounded self-supervised utility and beat matched random addressing, but did not beat FIFO significantly. EXP-010 did not establish consistent metric-geometry improvement. Capacity 64 is benchmark-selected; learned history eviction and DINO place consolidation remain rejected.

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
