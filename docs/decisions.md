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
