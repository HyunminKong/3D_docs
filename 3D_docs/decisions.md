# Research Decision Log

## D001 — Independent method, not a tttLRM extension

Date: 2026-08-20
Status: Accepted

Build a new TTT method and geometry head. Use foundation models only as backbones/priors. tttLRM remains a baseline and literature reference.

## D002 — VGGT as the first frozen backbone
Date: 2026-08-20
Status: Accepted for the first implementation

VGGT provides accessible dense features, camera estimates, and tracking evidence for controlled experiments. The method interface remains backbone-agnostic; D4RT is a later ablation/4D extension.

## D003 — Physical cross-episode revisit protocol

Date: 2026-08-20
Status: Accepted

Use explicit `A → B → A'` episodes with disjoint context/query frames and component-safe physical-overlap splits. Query frames measure future utility and cannot enter online TTT.

## D004 — Retire global compact state as the central memory object

Date: 2026-08-24
Status: Supersedes the compact-state design in `Research/revisit3d_v1_spec.md`

Global, slot, and learned vector updates collapsed or failed causal selectivity. The fast state will be spatially addressable 3D plasticity atoms.

## D005 — Utility, not episode identity, defines a correct retrieval

Date: 2026-08-24
Status: Accepted

Multiple past traversals can cover one physical region. Retrieval is evaluated by future-utility regret and negative transfer, not only by recovery of a designated manifest pair.

## D006 — Learned risk-aware utility routing

Date: 2026-08-24
Status: Accepted

Current geometry score is informative but unsafe as a hard gate. Train a candidate/current utility and risk head using future-frame outer supervision; allow rejection or soft mixing.

## D007 — Test-split closure after EXP-005

Date: 2026-08-24
Status: Accepted

The original six-episode test split was used once for the fixed dense-transport/online-utility probe. It is now exposed and must not guide further design. A future paper-scale benchmark requires a newly locked held-out test partition.

## D008 — Repository documents are the source of truth

Date: 2026-08-24
Status: Accepted

Chat discussions become official only when hypotheses, decisions, experiments, and the current state are updated in this repository and committed.

## D009 — EXP-006 uses a train-only, utility-conditioned pre-registered protocol

Date: 2026-08-25
Status: Accepted

Before atom/router training, the custom base geometry must pass an explicit predicted-depth/pose/confidence health gate. Atom meta-training uses the same five-candidate pool as routing and labels candidates by future utility rather than matched-episode identity. Discrete correspondence, inlier, Sim(3), and neighbor selection are detached under a first-order meta-gradient contract. Utility risk uses a train-calibrated neutral deadband. Any permitted model selection, including training length, occurs by grouped cross-validation within train; official validation is evaluated once and cannot select checkpoints or thresholds. The complete pre-registered protocol is `3D_docs/EXP-006 Implementation Brief.md` v2.2.

## D010 — Normalize VGGT confidence and group inference by overlap component

Date: 2026-08-25
Status: Accepted before EXP-006 training

A train-only preflight showed that FastVGGT depth confidence uses `1+exp(logit)` and occupies approximately `[1, 1.03]`; direct `[0,1]` clamping destroys all variation. EXP-006 therefore inverts `expp1` and uses train-only 5th/95th logit quantiles as the distillation target. Manifest graph inspection also showed 8/2/1 independent physical-overlap components in train/validation/closed-test, rather than 10/7/3 independent pairs. Cross-validation and bootstrap grouping use connected overlap components. The two-component validation can establish only descriptive feasibility, not paper-level inferential evidence.

## D011 — Track-loss pose health is teacher retention, not identity improvement

Date: 2026-08-25
Status: Accepted after the preserved Stage-0 v2.1 train-only diagnostic and before validation

The v2.1 cross-fit pose passed rotation, translation, scale, confidence, and gradient checks but its track loss was 1.098 times the identity-pose loss. A train-only counterfactual showed that the frozen VGGT teacher pose itself was worse than identity under the same objective (about 1.16 times on the directional mean). Identity removes parallax and is therefore a degenerate minimizer of this residual, not a valid pose baseline. EXP-006 freezes pose online and gates whether the custom pose preserves teacher-pose track behavior within 5%; identity remains a reported degeneracy diagnostic. The v2.1 result is preserved and not overwritten.
