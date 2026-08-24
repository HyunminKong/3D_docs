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

## D012 — Local transport bandwidth is estimated within each view

Date: 2026-08-25
Status: Accepted after the preserved Stage-1 v2.2 train-only diagnostic and before validation

The first predicted-transport preflight computed median 8-NN spacing after pooling every view in a segment-local point cloud. Because adjacent views repeatedly observe the same surface, cross-view near-duplicates collapsed the local bandwidth and caused valid physical revisits to fail the dimensionless residual gate: matched alignment validity was 0.70. A train-only counterfactual that changed only the scale estimator to within-view 8-NN raised matched validity to 1.00. EXP-006 v2.3 therefore defines local spacing over points from the same view while retaining a shared segment-local coordinate frame for Sim(3). The v2.2 failure is preserved; thresholds were not relaxed and validation remains unopened.

## D013 — Future-query state readout is appearance-only

Date: 2026-08-25
Status: Accepted before atom training and validation

The v2.3 train preflight additionally computed current-context→query Sim(3) as a diagnostic. It did not affect training, routing, thresholds, or candidate selection, but it conflicts with the pre-registered F10 rule prohibiting query geometry in alignment. That result is preserved as an invalid diagnostic rather than cited as evidence. EXP-006 v2.4 uses predicted geometry only for source→current transport. Future loss reads the current code at query tokens with a common visual-only transport; query features participate only in read-only prediction/outer loss, and query geometry never enters alignment, online TTT, routing features, or selection.

## D014 — Atom meta-objective protects absolute current-only quality

Date: 2026-08-25
Status: Accepted after the preserved v2.4 train-only objective failure and before validation

The registered v2.4 atom loss allowed gradient through `ell_current` inside `softplus(L_benefit-ell_current)`. Minimization could therefore raise current-only loss and manufacture a large relative candidate utility. The full-train refit exhibited exactly this failure: mean current-query loss rose from 0.0610 at initialization to 0.5838 (9.57×), while the reported best utility rose to 0.777. EXP-006 v2.5 normalizes current and candidate outer losses by the zero-code base query loss, directly minimizes `ell_current`, and detaches the current reference inside the relative margin. Train-CV checkpoints must first satisfy mean component current/base ≤1.05; utility selects only among safe checkpoints. The v2.4 result and checkpoint are preserved and cannot support H2-P.

## D015 — Reuse is a bounded residual after current TTT

Date: 2026-08-25
Status: Accepted after v2.5 train-only ablations and before validation

The safe v2.5 head improved the offline known-pose track objective in 19/20 train episodes, so the custom current TTT is healthy. The original candidate application remained unsafe: full geometry+appearance initialization had mean utility −0.399 and 82% harm. A train-only strength/application sweep showed that adding `0.10 * transported_atom` after current-only TTT changed geometry+appearance to mean utility +0.0103 with 10% harm. EXP-006 v2.6 therefore uses `clamp(z_current + 0.10*T(z_source),-1,1)` rather than adapting from the transported state. This keeps exactly one current-context TTT step and treats memory as a bounded residual. The same fixed strength applies to all transport ablations; validation cannot tune it.

## D016 — Out-of-fold estimates are authoritative for architecture decisions

Date: 2026-08-25
Status: Accepted after exact fold-specific Stage-1 refits

The v2.6 full-train refit reached current/base 0.389, while exact held-out-component evaluation was 0.865. Full-fit ablations on the same 20 training episodes therefore substantially overstate head quality and can reverse transport conclusions. EXP-006 architecture decisions use fold-specific heads whose held-out overlap component was absent from fitting. Full-fit results remain optimization diagnostics and are never overwritten.

## D017 — Visual transport carries the code; geometry is routing evidence

Date: 2026-08-25
Status: Supersedes predicted-geometry-primary transport in v2.6

Exact train OOF evaluation found visual local transport at +0.0162 mean future utility with 100% coverage and 2% candidate harm. Predicted geometry and geometry+appearance were similar in mean (+0.0161/+0.0162), but only 48% valid and had 6.25%/8.33% harm. H2-P is rejected in its deployable predicted form. The v2.7 primary path visually transports local codes. Predicted alignment validity, inliers, residual, and coverage remain observable router evidence and do not hard-mask an otherwise valid visual candidate.

## D018 — Adaptation regime, not physical identity, defines memory correctness

Date: 2026-08-25
Status: Strengthens D005

The designated matched traversal gave +0.0095 mean utility, while distant and foreign candidates averaged approximately +0.019 to +0.022 and the visual-candidate oracle reached +0.0323. A full observable grouped-OOF router selected the matched candidate in only 1/20 episodes. Memory records and future consolidation must therefore be evaluated and organized by causal adaptation utility/regime; scene identity, appearance similarity, and manifest pairing are priors or controls only.

## D019 — Risk-label diversity gates neural routing and validation

Date: 2026-08-25
Status: Accepted before neural router training or validation access

The leakage-safe visual candidate set contains 65 beneficial, 33 neutral, and 2 harmful examples. Both harmful examples occur in one episode and one overlap fold; leaving that fold out gives a risk-training partition with no harmful labels. A grouped risk classifier is therefore not identifiable on the current sample. EXP-006 validation remains unopened. The train benchmark must be expanded until harmful examples occur in more than one independent overlap component and every OOF training partition contains both benefit and harm before a neural risk result can be considered valid.

## D020 — One current TTT step plus memory is not replaceable by extra TTT

Date: 2026-08-25
Status: Accepted

Exact OOF evaluation found one current step/base 0.865. A second current step moved the ratio to 0.958 and produced −0.120 relative utility with 60% harm. In contrast, five-memory visual mean after one current step gave +0.0165 utility with no harm, and the oracle candidate gave +0.0323. The primary method keeps exactly one current step and adds a bounded memory residual; extra online optimization is a negative control, not the method.

## D021 — Expand train components without touching protected holdouts

Date: 2026-08-25
Status: Accepted and completed

The initial 8-component train set concentrated all two harmful visual candidates in one component, making grouped risk learning unidentifiable. Existing converted nuScenes data was audited by pose/location metadata only. Components touching the original validation or exposed test scenes were excluded, leaving 19 safe train components, 38 undirected overlaps, and 76 directional episodes. Original 14 validation and 6 test episodes are copied unchanged. Train/holdout scene intersection is empty. The expansion result is `benchmark_expansion_train_v27.json`; no held-out image or model output was accessed during construction.

## D022 — Remove predicted alignment from the primary router

Date: 2026-08-25
Status: Accepted after expanded exact OOF evaluation

On 19 components, visual transport reached +0.0180 mean candidate utility at 100% coverage. Geometry and geometry+appearance fell to +0.0152/+0.0164 at 70.8% coverage and higher harm. In Stage 2, adding predicted-alignment statistics to appearance plus adaptation-history routing changed utility only from +0.0280 to +0.0282 and did not reduce harm. Geometry-only selection was +0.0184 with 6.6% harm. Predicted alignment is therefore removed from the primary model and retained only as an ablation/diagnostic. The active observable statistics are visual-transport and current/source adaptation-history measurements.

## D023 — Lock the one-shot validation model before access

Date: 2026-08-25
Status: Accepted before EXP-006 validation cache/evaluation

The locked Stage-2 model is a train-only regularized utility regressor: current/candidate 64-D descriptors with difference/product, plus 20 visual/current/source adaptation-history scalars; train-only standardization; PCA-16; Ridge alpha 1; predicted-alignment scalars excluded. It selects the largest predicted utility only when that value is positive, otherwise returning current-only TTT. The code is visually transported and applied after exactly one current TTT step at fixed strength 0.10. On expanded train OOF it achieved +0.0280 selected utility, 82.9% benefit, 2.63% harm, and 0.0051 regret, beating visual mean (+0.0184, 3.95% harm), current objective (+0.0138, 2.63% harm), appearance similarity (+0.0159, 7.89% harm), and matched identity (+0.0147, 9.21% harm). Neural risk heads and predicted geometry are ablations, not the validation model. Validation is one-shot; its result cannot change features, PCA rank, ridge alpha, threshold, strength, candidate set, or application order.

## D024 — Register the v2.8 descriptive validation decision rule

Date: 2026-08-25
Status: Accepted before validation cache creation or model-output access

The v2.8 claim is supported on the one-shot validation only if the locked router satisfies all of the following without tuning: mean selected utility exceeds the fixed 0.01 deadband; mean utility exceeds visual mean, current-objective selection, appearance similarity, matched identity, and random-candidate expectation; no physical-overlap component has mean utility below −0.01; directional harm is no worse than visual mean; and accept rate is at least 0.20. Failure of any item is reported as a failed strict claim rather than repaired on validation. Because validation contains only two physical-overlap components, this decision is descriptive feasibility evidence and cannot serve as paper-level confirmation. The final model is fit on all train-only OOF feature/utility rows after architecture selection; its artifact hash is frozen before validation evaluation.

## D025 — Close EXP-006 and open train-only continual-bank feasibility

Date: 2026-08-25
Status: Accepted after the single locked validation evaluation

The locked router achieved +0.01785 mean validation utility, zero deadband harm, and +0.00503 regret, exceeding all D024 controls. All five checks passed without tuning. This closes EXP-006 as feasibility support for visual local-code reuse and utility ranking. It does not establish learned rejection: the router accepted all 14 episodes, and the two-component absolute bootstrap interval crosses zero. EXP-007 may now study causal bank capacity and consolidation on train only. EXP-006 validation is closed to all EXP-007 design/model selection, and the exposed test split remains prohibited.
