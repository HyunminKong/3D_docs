# Research Hypotheses

Status values are `Supported`, `Partially supported`, `Open`, or `Rejected`.

## H1 — Locality of reusable adaptation

Reusable streaming adaptation is better represented by spatially local fast codes than by a model-wide update, provided the local code is transported into the current token frame.

Status: **Supported for the tested FastVGGT depth/point setting.** Global and slot states failed context selectivity, whereas transported visual local codes improved all primary means in EXP-030/031 and again over current-only within every TUM sequence in EXP-035. Universal superiority across backbones, pose heads, and dynamic scenes is not established.

## H2-P — Predicted geometry as the update carrier

Predicted 3D alignment transports a reusable update better and more safely than appearance-only correspondence.

Status: **Rejected.** Predicted geometry and geometry+appearance transport did not improve over visual transport, reduced coverage, and increased harm. Oracle-coordinate EXP-005 remains an upper bound, not deployable evidence.

## H2-E — Predicted geometry as routing evidence

Predicted alignment quality is necessary primary evidence for candidate utility/risk even when geometry is not the update carrier.

Status: **Rejected as a primary input.** No-alignment observable routing generalized, while geometry-only routing was weaker and less safe. Alignment statistics remain diagnostic ablations.

## H3 — Online utility observability

Current/source descriptors, current-only loss changes, update statistics, and transport evidence predict the future utility of a past local correction without accessing future frames online.

Status: **Supported in the tested static-revisit setting.** EXP-029 learned a source-safe metric-utility address without online query access. The frozen address generalized in EXP-030/031 and retained positive sequence-balanced means in zero-shot TUM EXP-035. TUM acceptance was 100%, so cross-domain rejection calibration remains unsupported.

## H4-U — Learnable utility addressing and routing

A trainable candidate/current utility model can retrieve and apply more useful adaptation than identity, appearance/place similarity, current-loss heuristics, and matched random addressing.

Status: **Supported in the tested static-revisit setting.** The single factorized metric-utility Ridge address beat matched random and appearance means in development, untouched nuScenes terminal evaluation, and the three-sequence TUM descriptive transfer. This does not establish reliable per-sample harm rejection or transfer beyond the FastVGGT feature space.

## H4-R — Learnable negative-transfer risk

A separate risk classifier can generalize rejection of harmful memories across independent physical contexts.

Status: **Rejected for the tested classifier.** Neural risk AUROC was nontrivial, but hard routing did not improve selected harm over the compact utility router. Safety in the final model comes from utility selection, current-only fallback, and a fixed 0.10 residual—not a separately claimed risk classifier.

## H5 — Continual local-code consolidation

A causal, capacity-bounded memory can retain most of an unbounded utility-addressed bank's revisit benefit.

Status: **Partially supported for bounded deployment.** Reservoir-64 retained a positive metric-memory effect on the frozen EXP-031 location streams. It was not compared with an unbounded metric-addressed bank on that terminal test and has never significantly beaten FIFO. No reservoir-superiority or universal-capacity claim is made.

## H6 — Extension to dynamic 4D

The same utility-routed local-memory principle can attach to tracked dynamic points or motion-conditioned regions to improve reappearance and occlusion recovery.

Status: **Open; outside the completed static-revisit milestone.**

## H7 — Pareto-healthy offline plasticity

The aligned log-depth and aligned relative-depth meta objectives expose
different but jointly necessary geometry signals. A parameter-free common
descent update applied only during offline head meta-training can preserve
improvement in both, while deployment remains one-loss/one-step TTT.

Status: **Supported for the selected offline head.** EXP-026 exposed local
gradient conflict, EXP-027 localized AdamW rotation, and EXP-028's
parameter-free feasible-displacement safeguard achieved 100% realized common
descent while improving all three OOF geometry means. The frozen head retained
the full-system benefit in EXP-030/031. This is a training-health mechanism,
not the paper's standalone novelty claim or a general Pareto theorem.

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
