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

## D026 — EXP-007 starts with an exhaustive train-only causal utility table

Date: 2026-08-25
Status: Accepted before EXP-007 execution

EXP-007 does not begin by training a merge network. Stage 0 first computes, on expanded train only, the future utility and locked-router score of every unique observed context against every train current episode. Query frames are read only for offline utility labels. Causal simulations then expose a record only after its context is observed: within an event, A/B are written before A′ retrieval and the current A′ context is written afterward. Ten fixed pseudo-stream orders, appearance top-5 prefiltering, capacities {4,8,16,32,64}, and FIFO/reservoir/scene-latest/appearance-diversity controls are registered in `configs/EXP-007_causal_bank_v10.yaml`. These synthetic episode orders test memory mechanics, not real temporal generalization. A bounded policy is a promising consolidation direction only if capacity at most 32 retains at least 90% of unbounded-router utility without higher deadband harm.

## D027 — Full-train bank scaling is diagnostic; OOF banks cannot mix head coordinates

Date: 2026-08-25
Status: Accepted after EXP-007 v1.0/v1.1 train diagnostics

The v1.0 registered ratio gate mechanically passed for FIFO-8, but this is not consolidation evidence. Appearance top-5 recalled only 11–13% of the causal all-bank utility oracle, and the locked router's deadband harm increased to 14–20% as the candidate set grew. Small FIFO banks benefited partly by deleting distractors. Moreover, a global OOF stream cannot mix atoms produced by different fold-specific heads because their learned code/key coordinates are not identical. Authoritative v1.2 evidence therefore runs a separate causal stream inside each held-out fold, using one fold head and a router trained on the other folds. Only contexts observed within that fold stream can be written. Full-train v1.0/v1.1 results are preserved as scaling diagnostics, not architecture-selection evidence.

## D028 — Utility-history consolidation must beat the scene baseline

Date: 2026-08-25
Status: Accepted before EXP-007 v1.3 execution

Fold-local OOF streams show that scene-latest capacity 8 improves over unbounded all-write top-5 by +0.00466 utility with a paired component-bootstrap CI [0.00189, 0.00725], while reducing the maximum bank from 83 to 8. This establishes bounded-memory feasibility but not adaptation-aware novelty because the benchmark contains repeated scene contexts. EXP-007 v1.3 therefore compares fixed online statistics at capacities 4/8/16: mean predicted utility history, selected-memory realized-utility UCB, delayed top-5 counterfactual utility, and a hybrid. A noncausal future-coverage coreset is an upper bound only. At capacity 8, a utility-history method is admitted as the primary consolidation mechanism only if it exceeds scene-latest mean utility without higher deadband harm. Otherwise scene identity remains a strong baseline and learned consolidation requires a richer stream/benchmark rather than an embellished mechanism.

## D029 — Calibrate acceptance for candidate-set size and causal history

Date: 2026-08-25
Status: Accepted before EXP-007 v1.4 execution

At capacity 8, hybrid history increased OOF mean utility from scene-latest's 0.02615 to 0.02808 but increased deadband harm from 6.32% to 6.71%, so D028 did not pass. The frozen EXP-006 router accepted essentially every causal-bank event and was trained on fixed K=5 pools, not on the max-score distribution induced by persistent banks. EXP-007 v1.4 keeps hybrid eviction fixed and trains only a compact acceptance calibrator from current-observable candidate-score distribution, bank size, current objective, and selected-record causal history. Evaluation is leave-one-physical-component-out across all 19 groups; all pseudo-order copies of a component stay in one fold. The primary gate accepts only when calibrated utility exceeds the existing 0.01 utility deadband. It must beat scene-latest mean utility without higher harm and retain at least 20% acceptance. Query utility is the target and delayed history only; it never enters same-event gate features.

## D030 — Replace pointwise max selection with a set-normalized utility reranker

Date: 2026-08-25
Status: Accepted before EXP-007 v1.5 execution

The v1.4 gate failed: it retained 6.71% harm, and the raw winning router score was negatively correlated with its realized utility (Spearman −0.457). This is a winner's-curse/list-distribution failure, not evidence that causal history is useless. EXP-007 v1.5 freezes the hybrid capacity-8 bank trajectory and represents every appearance-top-5 candidate by raw router/current/appearance values, within-set normalized values and ranks, and that record's past-only utility/prediction history. A leave-one-component-out Ridge utility reranker is primary; pairwise Ridge and non-set pointwise features are controls. Same-event query utility is the target only. The primary applies a memory only above the fixed 0.01 deadband and must exceed scene-latest utility without higher harm. This off-policy test may authorize an on-policy rerun but cannot by itself establish a final recurrent bank.

## D031 — Scene is a consolidation bucket, not the memory correctness label

Date: 2026-08-25
Status: Accepted before EXP-007 v1.6 execution

The v1.5 set-normalized reranker raised mean utility but retained 6.71% harm and failed the scene-latest safety requirement. Further pointwise/listwise gate complexity is not justified on the current data. EXP-007 v1.6 instead keeps the empirically safe one-record-per-scene bucket, replaces its atom with the latest observed context, transfers causal utility/prediction history within that bucket, and uses adaptation history rather than FIFO age to evict across scene buckets at capacity 8. Scene identity is therefore a coarse redundancy constraint, not a claim that the matched place is the correct update. Predicted, selected-realized, delayed-top-5, and hybrid history variants are fixed. If none beats scene-latest mean utility without higher harm, learned continual consolidation is not supported by this benchmark and H5 must remain partial/open pending richer streams.

## D032 — Replace oracle scene buckets with OOF-calibrated visual buckets

Date: 2026-08-25
Status: Accepted before EXP-007 v1.7 execution

The oracle-scene delayed-top-5 history policy passed D031 at 0.02674 utility and 6.18% harm, with a small positive paired component interval over scene-latest. Manifest scene IDs are unavailable at deployment and remain an oracle grouping baseline. EXP-007 v1.7 learns only a scalar cosine threshold for same-bucket decisions from the other atom folds, optimizing balanced accuracy, then applies that threshold online to fold-held-out descriptors. Ground-truth scene labels never enter runtime writes, retrieval, eviction, or routing. Visual-bucket predicted-history and delayed-top-5 variants keep capacity 8 and all other v1.6 settings fixed. A deployable bucket is promising if it beats appearance-diversity capacity 8 without higher harm and retains at least 90% of the oracle-scene utility.

## D033 — Consolidation keys must preserve token-set correspondence

Date: 2026-08-25
Status: Accepted before EXP-007 v1.8 execution

The v1.7 pooled-key threshold failed because held-out same-scene precision was only about 5–10%, producing hundreds of false merges and 84–85% oracle-scene utility retention. EXP-006 had already shown that per-token visual transport succeeds while global pooled descriptors are weak. EXP-007 v1.8 therefore computes deployable token-set statistics from the fold head: pooled cosine, bidirectional nearest-token means, lower-tail coverage, mutual-nearest coverage, and high-similarity quantiles. A balanced logistic bucket classifier is trained on other folds' scene-equality labels and evaluated on the held-out fold; runtime receives only token statistics and predicted probability. Capacity, history policies, and success criteria remain identical to v1.7. Failure closes the current H5 implementation as partial and redirects future work to a stronger place-recognition/consolidation backbone rather than further threshold tuning.

## D034 — Separate frozen consolidation features from learned transport features

Date: 2026-08-25
Status: Accepted before EXP-007 v1.9 execution

The learned atom-key token classifier reached only 0.654 OOF AUC and failed D033, suggesting that meta-training for local code transport does not preserve traversal-level place separability. One final control uses the frozen VGGT feature tokens projected by the already fixed train PCA, while retaining the identical token-set classifier, bank, and success criteria. If this frozen representation also fails, EXP-007 closes without a deployable consolidation key and the next architecture must add a separately trained/frozen place-recognition encoder such as DINOv3/AnyLoc rather than reuse either plasticity or VGGT reconstruction tokens.

## D035 — Correct the frozen-key control with fold-local PCA

Date: 2026-08-25
Status: Accepted before EXP-007 v2.0 execution

The v1.9 frozen-token control mechanically passed its registered point-estimate gate, but the PCA projection had been fit once on all train episodes. This does not access validation/test or labels, yet it is transductive with respect to the held-out OOF fold and is too weak a basis for an architecture decision. EXP-007 v2.0 is a leakage-correction control: for each held-out atom fold, it fits a 64-D PCA using only context tokens from the other folds, then keeps the same token-set statistics, crossfit logistic classifier, capacity-8 bank, policies, threshold, orders, and success criteria. No v1.9 result is used to tune these settings. The frozen consolidation key is provisional only if the v2.0 point-estimate gate reproduces; statistical uncertainty and real-stream generalization remain separate requirements.

## D036 — A consolidation key must beat a merge-rate-matched permutation null

Date: 2026-08-25
Status: Accepted before EXP-007 v2.1 execution

The strict crossfit frozen key reproduced the point-estimate gate at 0.02618 utility, 8.03% harm, and 97.9% oracle-scene retention, but its same-scene AUC was only 0.650, 773 of 1,839 merges crossed scene labels, and the component bootstrap interval versus appearance diversity included zero. This leaves a capacity-regularization explanation: arbitrary merging may simply remove router distractors. EXP-007 v2.1 therefore shuffles the OOF bucket probabilities among context pairs within each held-out fold, preserving each fold's score distribution and the fixed 0.5 threshold while destroying the learned pair association. The primary predicted-history key is supported only if its observed mean utility exceeds at least 95% of 1,000 matched permutations and does not exceed appearance-diversity harm. This diagnostic uses train only and cannot establish real-stream generalization.

## D037 — Close EXP-007 with a dual-address, bounded-memory architecture

Date: 2026-08-25
Status: Accepted after EXP-007 v2.1 execution

The strict frozen-token predicted-history bank achieved 0.02618 utility, 8.03% harm, 97.9% oracle-scene retention, and capacity 8. Its utility exceeded 99.4% of 1,000 fold-matched probability permutations (p=0.00699), so the key contributes beyond arbitrary memory shrinkage. Harm was not lower than the permutation-null mean, same-scene AUC was 0.650, and all streams were pseudo-orders. EXP-007 therefore partially supports H5 and selects a dual-address architecture: learned local keys for code transport, separately frozen token-set keys for consolidation/prefiltering, past-only predicted utility history for retention, and the bounded residual—not bucket similarity—as the present damage-control mechanism. Capacity 8 and the frozen VGGT key are provisional benchmark choices. EXP-008 must test unique writes in true capture-time order before any new holdout or larger learned bank is authorized.

## D038 — EXP-008 first corrects pseudo-order with true capture time

Date: 2026-08-25
Status: Accepted before EXP-008 Stage-0 execution

Before adding a place-recognition backbone or retraining a bank-aware router, EXP-008 reuses the locked train-only OOF utility table and strict crossfit bucket scores in actual nuScenes capture-time order. Every unique context is written once when its final context frame has been observed; a target is evaluated immediately before its own write. Duplicate target episodes are collapsed only if all candidate utilities and predictions are exactly identical. The primary frozen-bucket predicted-history capacity-8 bank must exceed appearance-diversity utility without higher deadband harm and retain at least 90% of oracle scene-latest utility. This stage does not access validation/test and cannot establish paper-scale generalization; it decides whether the dual-address bank survives a real chronology correction.

## D039 — True-time consolidation must also beat a matched merge null

Date: 2026-08-25
Status: Accepted before EXP-008 Stage-1 execution

Stage 0 passed strongly: the frozen-bucket predicted-history bank reached 0.02650 utility and 5.63% harm versus appearance diversity's 0.02387/7.04%, with a 19-component bootstrap CI [+0.00019, +0.00486]. It also exceeded the oracle scene-latest grouping baseline. However, 51 of 85 merges crossed scene labels, so compression itself remains a possible explanation. Stage 1 shuffles the strict OOF bucket probabilities within fold 1,000 times, preserving the score distribution, threshold, timestamp order, write count, capacity, and policy. The observed primary is attributed to the consolidation key only if it exceeds at least 95% of matched-null utilities and keeps no more harm than appearance diversity.

## D040 — True-time feasibility passes; new scenes are now mandatory

Date: 2026-08-25
Status: Accepted after EXP-008 Stage-1 execution

The true-time primary exceeded 96.1% of 1,000 matched probability permutations (one-sided p=0.03996) while keeping lower harm than appearance diversity. Together with the positive component interval, this closes the chronology correction and selects the dual-address bounded bank for the static-revisit milestone. Further tuning on the 76 expanded-train episodes is prohibited because the remaining uncertainty is independent-scene generalization, not another threshold. EXP-009 must blacklist every scene used by EXP-001–008 and construct a new component-disjoint benchmark from untouched nuScenes scenes using pose/location metadata only. No new learned bank-aware router, DINO/AnyLoc selector, or validation evaluation is authorized before that manifest is frozen.

## D041 — Build the new benchmark from metadata before converting images

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-0 execution

EXP-009 first inventories all unused nuScenes trainval scenes using only official camera/ego pose, timestamp, calibration, log location, and scene metadata. Every scene present in prior converted roots or manifests is blacklisted conservatively. Within each location, scene pairs receive an overlap edge at 2.0 m camera-center distance; connected overlap components are indivisible split units. A fixed deterministic greedy procedure targets 70/15/15 undirected-edge balance within each location. No image is opened and no model signal is available. If graph connectivity prevents credible independent splits, the protocol must be revised from metadata alone before any new scene is converted or encoded.

## D042 — Freeze all healthy unseen overlap edges before feature extraction

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-1 execution

The metadata inventory found 1,368 unseen undirected edges in 65 components. The fixed split has 2,268/234/234 directional train/validation/test episodes, zero scene overlap, zero component overlap, at least 86 scenes in each holdout, and all four locations in each holdout. The 304-scene giant component is confined to train. This is healthy enough to freeze without resampling. Stage 1 converts every edge in both directions using the established eight-context/four-query segment construction. All health and leakage assertions must pass before any scene conversion, feature extraction, or model training. The new validation and test remain unopened after manifest creation.

## D043 — Metadata conversion does not authorize holdout feature access

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-2 execution

The frozen manifest passed every D042 check and is immutable. EXP-009 may now generate converted camera metadata for all 636 manifest scenes because this operation reads official poses, intrinsics, timestamps, and image paths only; it does not decode pixels or run a model. The audit must match exactly 454 train, 86 validation, and 96 test scenes. After conversion, any image loading, feature extraction, PCA fitting, plasticity training, router fitting, or model selection remains train-only until an explicit validation lock decision is recorded.

## D044 — Choose the consolidation backbone on a balanced train-only key pilot

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-3 selection

Full geometry caching is deferred until the long-term consolidation representation is selected. Stage 3 samples at most 64 positive undirected train edges per location by a fixed hash, retaining at least 40 per location, and creates one strict same-location non-overlap negative per positive using metadata bounding-box separation of at least 20 m. Four fixed context views are used. VGGT reconstruction tokens and the locally available frozen DINOv2 ViT-L/14 representation will be compared under the same leave-one-location-out pair/retrieval protocol. Validation/test pixels remain unopened. DINOv2 is a consolidation-only candidate; selecting it would not replace the VGGT geometry backbone or the learned local transport key.

## D045 — DINOv2 requires a meaningful cross-location margin

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-4 feature extraction

The Stage-3 pilot contains 225 positive and 225 strict negative pairs across 247 train scenes. Stage 4 extracts frozen four-view sets: per-view mean patch tokens from VGGT and per-view normalized CLS tokens from locally cached DINOv2 ViT-L/14. A shared eight-statistic token-set classifier is evaluated leave-one-location-out. DINOv2 becomes the long-term consolidation representation only if OOF ROC-AUC is at least 0.03 above VGGT and every held-out location reaches at least 0.70 AUC. Otherwise the current VGGT representation remains. This decision affects only long-term bucketing/prefiltering; VGGT remains the geometry backbone and the learned plasticity key remains the transport address.

## D046 — Separate reconstruction, transport, and consolidation representations

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-4 execution

DINOv2 achieved 0.936 leave-one-location-out OOF AUC versus VGGT's 0.744, a +0.193 margin; its weakest held-out location reached 0.900. The registered gate passed. The selected architecture now has three explicit representation roles: frozen VGGT dense tokens for reconstruction and current TTT evidence, a learned local key/code for token-level adaptation transport, and frozen DINOv2 view tokens for long-term consolidation and candidate prefiltering. This is not redundant backbone stacking: the experiment shows reconstruction tokens and place-compatible memory addresses have materially different invariances. DINOv2 is not a safety gate and has not yet established causal adaptation utility. EXP-009 Stage 5 must verify locked plasticity/reuse transfer on new train scenes before training a bank-aware router or opening validation.

## D047 — Test locked local reuse before attributing benefit to DINOv2

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-5 execution

One direction of each Stage-3 positive edge forms a 225-episode train-only geometry pilot. Stage 5 first transfers the fully locked EXP-006 local plasticity path without retraining: custom VGGT-token head, one current TTT step, visual code transport, fixed 0.10 residual, K=5 candidate pool, and locked linear utility router. The gate requires a healthy current/base ratio, visual-mean utility above 0.01 with under 10% harm, at least 10 physical components, and router utility above visual mean without higher harm. This ordering prevents a stronger place key from hiding failure of the underlying adaptation. Validation/test pixels remain unopened.

## D048 — Retain local plasticity; recalibrate utility routing by nested components

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-6 execution

Stage 5 separated transfer outcomes. Current TTT was healthy (current/base 0.682), visual reuse remained useful (+0.01227), and the oracle candidate reached +0.02600, so the local head/transport mechanism transfers. The old router raised utility to +0.01695 but increased harm to 14.22% versus visual mean's 11.11%, failing safety. Stage 6 therefore keeps every adaptation mechanism fixed and retrains only the same PCA-16 Ridge utility model. Evaluation is outer leave-one-component-out over 25 physical components; each outer threshold is calibrated solely from inner component-OOF predictions to maximize utility under the outer-train visual-harm bound and at least 20% acceptance. Success requires higher utility than visual mean, no higher harm, nontrivial acceptance, and a positive paired component-bootstrap lower bound.

## D049 — Stop threshold tuning after the nested interval misses zero

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-6 execution

The nested component router reached +0.01475 utility with 10.67% harm and 83.56% acceptance, improving on visual mean's +0.01227/11.11%. Its paired component-bootstrap interval was [-0.00014, +0.00465], so the registered positive-lower-bound check failed narrowly. Further thresholds on the same 225-episode candidate table would be post-hoc tuning and are prohibited. The fixed PCA-16 Ridge model and nested calibration remain the router diagnostic for the next candidate-set experiment.

## D050 — Evaluate DINOv2 by causal top-K adaptation utility

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-7 feature extraction

Place-pair AUC alone does not prove that a consolidation address retrieves reusable plasticity. Stage 7 replays every unique A/B/A′ train context once in true capture-time order and evaluates unique A′ targets before their own write. Frozen DINOv2, the learned VGGT-side transport key, FIFO, and deterministic random each retrieve K=5 from the same causal history. The primary outcome is oracle utility within each candidate set; the existing nested router is secondary. DINOv2 is causally supported only if its oracle top-K and routed utility both exceed the VGGT key and routed harm is no higher. Query frames remain offline labels only, and EXP-009 validation/test pixels remain unopened.

## D051 — DINOv2 beats the transport key but misses the causal safety gate

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-7 execution

In the 557-context true-time replay, DINO top-K increased oracle utility from the VGGT transport key's 0.01958 to 0.02412 and routed utility from 0.00889 to 0.01208. Both paired 25-component intervals were positive. Routed harm was 10.55% versus 10.09%, so the registered gate failed by one of 218 targets. A stronger warning is that one deterministic-random top-K achieved 0.02632 oracle and 0.01663 routed utility, and DINO retrieval score had Spearman -0.038 with realized utility. DINO remains a place-compatible representation, but it is not yet accepted as a plasticity-utility address.

## D052 — Require a matched random retrieval null before designing the bank

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-8 candidate evaluation

Stage 8 samples a fixed uniform panel of at most 64 memories from each target's exact causal history, evaluates their locked future utility, and simulates 2,000 random K=5 policies. DINO is causally supported only if both oracle and routed mean utility exceed 95% of this matched null, their component-bootstrap differences over per-target random expectation have positive lower bounds, and routed harm is no higher than the null median. No DINO score, router threshold, or transport mechanism is changed. If the gate fails, generic place recognition is retired as the consolidation objective and the next key must be trained directly on future adaptation utility using train-only observable context pairs.

## D053 — Retire generic place compatibility as the plasticity address

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-8 execution; supersedes D046 for consolidation

The matched random K=5 null averaged 0.02642 oracle and 0.01614 routed utility, exceeding DINO's 0.02412/0.01208 in all 2,000 aggregate repetitions (one-sided p=1.0). DINO-minus-random component intervals were strictly negative for both outcomes. Place recognition is therefore not the correct optimization target for reusable adaptation, even though DINO remains highly predictive of physical overlap. DINO may remain one observable feature or baseline, but it is no longer the central consolidation address. Reconstruction remains VGGT-based and local code transport remains the learned per-token visual key.

## D054 — Test utility-supervised prefiltering before a scalable memory key

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-9 fitting

Stage 9 uses the fixed Stage-8 causal panels and future utility labels to cross-fit a cheap pre-transport pair scorer. Its deployable inputs are frozen DINO pair statistics, pooled learned transport descriptors, and current/source TTT objective histories; query information and post-transport statistics are prohibited. A fixed StandardScaler→Ridge(alpha=1) model over all 274 observables is primary, with four registered feature ablations. It must beat the per-target matched-random expectation with positive component intervals for both oracle top-K and unchanged-router utility, show positive OOF association, and not exceed random-median harm. Success authorizes distillation to a scalable utility-conditioned address; failure means the available pre-transport context does not identify reusable updates.

## D055 — Utility-conditioned prefiltering is feasible; DINO-only is anti-predictive

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-9 execution

The all-observable Ridge prefilter passed at 0.03253 oracle top-K and 0.02012 routed utility versus random 0.02642/0.01614, with strictly positive 25-component intervals. DINO-only had OOF Spearman -0.187, while transport descriptors reached +0.281 and adaptation histories +0.379. The transport-only scorer is especially attractive: it achieved 0.03241 oracle, 0.02039 routed utility, and 8.72% harm, and its linear descriptor interaction can be represented exactly as maximum inner product search. This supports the core claim that adaptation utility, not place identity, should supervise the long-term address.

## D056 — Correct source-memory entity leakage before locking the address

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-10 fitting

Target-component OOF does not exclude a memory source from appearing in another target's training pairs. The audit found 147 to 4,002 held-location source rows in the corresponding naive training folds. Stage 10 therefore leaves out an entire location and removes every training row whose target or source belongs to that location. The transport-descriptor Ridge is the registered primary because it is both strong and exactly factorable into a 64-D MIPS query/source score; DINO-only, history-only, and all-observable models are ablations. Positive pooled and each-location association plus random-null utility and harm gates are required before fitting a deployable artifact.

## D057 — Accept the factorized utility address after source-safe transfer

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-10 execution

With held-location targets and sources excluded from training, the transport-descriptor scorer retained positive association in all four locations (0.182–0.235), 0.03149 oracle top-K utility, 0.01933 routed utility, and 9.17% harm. Its random-relative component intervals remained positive. Because its linear features are `[c, s, c-s, c*s]`, the score can be rearranged exactly into a current-conditioned 64-D maximum-inner-product query against each stored source descriptor. This utility-MIPS form is the provisional long-term address. A capacity policy must still pass a causal replay before validation is opened.

## D058 — Test history retention at capacity 8 before validation

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-11 execution

The strong history-only utility signal may be useful for retention even though the transport descriptor is the retrieval address. Stage 11 runs independent true-time streams per location and trains both scores with the held location entirely excluded. The primary bank keeps the eight records with highest source-only adaptation-history priority and retrieves five by utility-MIPS. Unbounded, FIFO-8, and deterministic reservoir-8 are controls. History retention is accepted only if it retains at least 90% of unbounded routed utility, beats FIFO and reservoir, has no greater harm than FIFO, and has a positive component interval over FIFO. A failed learned policy is not repaired; the safest simple policy becomes the validation bank.

## D059 — Use reservoir retention; reject learned history eviction

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-11 execution

History capacity 8 achieved 0.01747 routed utility but 12.84% harm, failed against reservoir, and had a FIFO-difference interval crossing zero. Deterministic reservoir capacity 8 achieved 0.01768 utility and 8.72% harm, exceeded FIFO by a component interval [0.00078, 0.00590], and descriptively exceeded the unbounded addressed bank. The final static-revisit bank therefore uses deterministic reservoir retention. Adaptation history remains useful evidence but is not a reliable eviction objective in the present model.

## D060 — Lock the deployable artifact and one-shot validation gate

Date: 2026-08-25
Status: Accepted before EXP-009 validation pixel access

Stage 12 fits the transport-descriptor Ridge and final PCA-16 Ridge router on all permitted train pairs, compiles the address exactly to 64-D MIPS, and records the artifact hash. The validation pilot contains one metadata-selected direction for each of 117 unseen validation overlaps across 17 components and four locations. The locked method is one current TTT step, reservoir capacity 8, utility-MIPS K=5, visual local-code transport, fixed 0.10 residual, and the fixed router/threshold. One-shot validation requires healthy current TTT, routed utility above 0.01, at least 90% unbounded retention, superiority to FIFO without more harm, superiority to a matched random address, and a positive component interval over random. No validation result may change these choices.

## D061 — Validation supports the address but rejects capacity 8

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-13 one-shot validation

The fixed utility-MIPS address with an unbounded bank reached 0.02642 routed utility and 5.83% harm on 103 unique targets across 17 unseen components. Reservoir-8 reached 0.01937 and beat FIFO-8, but retained only 73.3% of unbounded utility; its harm was 8.74%, and its random-address interval crossed zero. The failure is therefore attributed to the provisional capacity-8 assumption, not to local adaptation or the utility address. Capacity 8 is rejected as a universal constant. The validation split is now exposed for explicit capacity selection; the 22-component test split remains untouched.

## D062 — Select the smallest passing capacity without changing the model

Date: 2026-08-25
Status: Accepted before EXP-009 Stage-14 validation reuse

Stage 14 changes only the reservoir/FIFO capacity over the fixed set {8,16,32,64}. Atom weights, TTT step, utility-MIPS coefficients, router, threshold, K=5, residual 0.10, and stream partition remain frozen. For each capacity, random K=5 addressing is evaluated inside the identical reservoir bank. The smallest capacity is selected only if it retains at least 90% of unbounded routed utility, has no more harm than unbounded, beats FIFO and random addressing, and has a positive component interval over random. No passing capacity means the bounded-bank claim is withheld from test.

## D063 — Select capacity 64 on validation

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-14 execution

Capacities 8, 16, and 32 failed the fixed retention/safety gate. Capacity 64 was the smallest passing choice: 0.02647 routed utility, 100.2% retention of unbounded, 5.83% harm, and a positive component interval [0.00026, 0.01097] over random addressing. FIFO-64 reached 0.02606. The selected capacity is benchmark-dependent rather than a universal constant. All learned weights, thresholds, K, and residual strength remain those locked before validation.

## D064 — Authorize one final 22-component test after hash freeze

Date: 2026-08-25
Status: Accepted before any EXP-009 test pixel access

Stage 15 freezes one canonical direction for each of 117 test overlaps using metadata only, spanning 22 components and four locations. The final artifact is a byte-serialized copy of the frozen address/router with only capacity changed from 8 to the validation-selected 64; its hash must be committed before caching test pixels. Stage 16 is one-shot and terminal. It requires healthy current TTT, routed utility above 0.01, at least 90% unbounded retention, no more harm than unbounded, superiority to FIFO and same-bank random addressing, and a positive component interval over random. The test result cannot be used for further method selection.

## D065 — Accept utility-addressed bounded reuse; do not claim reservoir superiority

Date: 2026-08-25
Status: Accepted after EXP-009 Stage-16 terminal test

The pre-locked reservoir-64 system passed all eight terminal gates on 104 unique targets across 22 unseen physical components. Routed utility was 0.02088 with 3.85% harm, versus 0.01602 for same-bank random addressing. The paired component difference was 0.00475 with 95% CI [0.00086, 0.00924], and reservoir retained 100.98% of unbounded utility. This supports the final static-revisit claim: observable utility addressing retrieves reusable local TTT experience in a causal bounded bank.

Reservoir and FIFO were nearly identical (0.02088 versus 0.02068), and their difference interval [-0.00013, 0.00059] crossed zero. The paper must therefore not claim that reservoir is the learned or uniquely superior consolidation strategy. Generic DINOv2 place addressing, learned history eviction, and universal capacity assumptions remain rejected. EXP-009 test is terminal and closed to all further tuning; the next milestone must use new development data for absolute reconstruction, resource, pose, or dynamic-4D evaluation.

## D066 — Freeze a compact CVPR-first paper scope

Date: 2026-08-25
Status: Accepted before EXP-010 execution

The first publication target is one compact static-revisit paper, not a joint depth/pose/tracking/4D architecture. The primary method consists of a local plasticity atom and one coarse-to-fine utility-retrieval mechanism; fixed-capacity reservoir storage is an implementation constraint. DINO retrieval, predicted-geometry transport, neural risk heads, learned eviction, pose adaptation, extra TTT steps, and dynamic state are excluded. The generic phrases “TTT memory” and “retrievable gradients” are not novelty claims because tttLRM/ZipMap/Mem3R and ReGrad create direct prior-art collisions. The defensible contribution is spatially transported local adaptation addressed by causal future geometry utility.

EXP-010 first tests whether the already locked method improves sparse-LiDAR geometry. This secondary endpoint may change the paper claim but may not change any EXP-009 weight, threshold, residual, address, K, or capacity. A later train/validation-only refit must simplify feasibility-stage meta-losses before the final paper model is accepted.

## D067 — Stop memory expansion after the absolute-geometry bridge fails

Date: 2026-08-25
Status: Accepted after EXP-010 Stage A

The frozen full model significantly improved per-view aligned AbsRel over current-only TTT, but worsened SILog, aligned RMSE, and same-ray 3D endpoint error. The registered absolute-geometry gate therefore failed. This prevents a broad reconstruction or point-cloud claim and pauses random/FIFO/router expansion: more memory experiments cannot repair a misaligned online/meta objective.

The next experiment must remain train-only and simplify rather than enlarge the method. It will compare the existing metric-space 3D track consistency objective with a single symmetric track-reprojection objective and a small one-step-size sweep. No additional online loss is authorized. Only an objective that improves SILog, aligned AbsRel, and 3D endpoint error together may trigger atom/router retraining. EXP-009 test remains closed to tuning.

## D068 — Freeze one 3D-track loss and a smaller TTT step

Date: 2026-08-25
Status: Accepted after EXP-011 Stage 0 and before validation LiDAR access

The train-only audit covered 218 unique targets and 25 components. A single absolute 3D frozen-track consistency loss at `eta=0.0125` improved the means of SILog, median-aligned AbsRel, and same-ray 3D EPE; the component-bootstrap AbsRel interval was strictly positive. Adding smoothness and code regularization was numerically immaterial, so the paper online objective is simplified to one loss. The symmetric reprojection alternative improved SILog/EPE at the smallest step but worsened AbsRel, and larger steps exposed the same overshoot seen in EXP-010.

The loss, one-step count, and `eta=0.0125` are now frozen for one validation audit. Validation may confirm or reject this objective but may not tune it. EXP-009 test remains terminal and unavailable.

## D069 — Accept the minimal objective for the paper refit

Date: 2026-08-25
Status: Accepted after EXP-011 one-shot validation

The frozen one-loss, one-step objective passed on all 103 valid validation targets over 17 components. SILog, aligned AbsRel, and same-ray 3D EPE improved by 0.474%, 0.959%, and 0.992% respectively; the component-bootstrap EPE interval was strictly positive. This independently confirms the train-stage direction while retaining one online loss and one step-size hyperparameter.

EXP-012 may now retrain the local atom and utility address/router on train data using this fixed objective. It must simplify the feasibility meta-objective rather than add heads or losses. Validation is exposed for the registered paper-model gate; EXP-009 test remains closed and a new external component-disjoint benchmark is required for final evidence.

## D070 — Test a one-objective atom before refitting memory

Date: 2026-08-25
Status: Accepted before EXP-012 Stage 0 execution

The paper candidate freezes the train-PCA visual transport key and trains only the 157,121-parameter local-code decoder. Its entire meta-objective is the equal average of current-only and matched-reuse future 3D-track loss. Five feasibility terms—key contrastive, harmful-code neutralization, code centering, depth smoothness, and code norm—are removed. Three training epochs are fixed without checkpoint selection.

Evaluation is five-fold over the immutable 25 physical overlap-component IDs. The minimal atom must preserve current TTT, exceed 0.5% matched reuse utility with at most 20% harm, and beat the distant within-episode source with a positive component-bootstrap lower bound. No memory scorer is fitted unless this gate passes.

## D071 — Reject matched identity, retain one utility-selected meta-objective

Date: 2026-08-25
Status: Accepted after EXP-012 Stage 0A and before Stage 0B

Stage 0A improved current future loss but failed both reuse gates: matched utility was +0.270%, distant utility was +0.312%, and their component interval crossed zero. This is not decoder collapse; it falsifies the assumption that metadata-matched episode identity supplies the correct positive adaptation.

Stage 0B changes only the offline discrete target selection. It evaluates five train-only source codes and minimizes the future loss of the best one, averaged equally with current-only future loss. Architecture, single 3D-track loss, `eta=0.0125`, `alpha=0.10`, fixed PCA key, optimizer, and three epochs remain unchanged. No auxiliary loss or new head is introduced. The Stage-0A result is immutable and retained.

## D072 — Test one relative-utility term under unchanged gates

Date: 2026-08-25
Status: Accepted after EXP-012 Stage 0B and before Stage 0C

Stage 0B established positive candidate mean and significant oracle selection headroom, but its +0.465% oracle utility missed the 1% gate and its 30.0275% harm narrowly exceeded the 30% limit. Equal averaging improves absolute candidate loss without explicitly requiring reuse to beat current adaptation.

Stage 0C replaces that average with current normalized future loss plus an unweighted softplus ranking of best reuse against stop-gradient current loss. This is the only independently motivated feasibility term restored. All architecture, data, candidates, optimization settings, and Stage-0B gates remain fixed. Failure stops the compact atom family rather than triggering further loss accumulation.

## D073 — Stop the frozen-key family; align the existing key with utility

Date: 2026-08-25
Status: Accepted after EXP-012 Stage 0C and before EXP-013

Stage 0C passed every gate except oracle magnitude: current/base was 0.8340, mean candidate utility +0.208%, harm 27.03%, and oracle-minus-mean CI strictly positive, but oracle utility was only +0.472%. The frozen-PCA-key compact family is stopped as registered.

EXP-013 makes the existing 64-D key projection trainable under the same future-utility ranking objective. This adds no parameter tensor, module, loss, or weight; it replaces task-agnostic PCA transport with utility-supervised transport while retaining PCA initialization. All EXP-012 Stage-0C gates remain unchanged. Failure ends local-key redesign for this paper.

## D074 — Stop utility-key training; audit the pre-selected 1000-step budget

Date: 2026-08-25
Status: Accepted after EXP-013 and before EXP-014

EXP-013 failed decisively: mean candidate utility became negative and harm rose to 45.38%. End-to-end key redesign is stopped. A separate audit found that every new minimal fold had only 483–576 training updates, whereas the pre-existing EXP-006 v2.7 cross-fit selected 1000 over 500 steps and improved oracle utility from 1.323% to 3.855%.

EXP-014 therefore restores the safer frozen PCA key and repeats the minimal relative-ranking model with exactly 1000 updates. This uses an optimization budget selected before EXP-012 and changes no method component, loss, inference hyperparameter, or gate. Failure triggers a paper-scope pivot rather than further atom variants.

## D075 — Make the core three-term utility objective the final atom variant

Date: 2026-08-25
Status: Accepted after EXP-014 and before EXP-015

EXP-014 raised oracle utility to +0.9205% and passed every gate except the fixed 1% magnitude threshold. Together with the failure of absolute-reuse-only Stage 0B, this supplies a factorial reason to combine absolute best-reuse quality and reuse-versus-current ranking rather than restore unrelated auxiliary losses.

EXP-015 uses current loss, best-reuse loss, and their unweighted softplus ranking. All are evaluations of the same single 3D-track signal; key contrastive, neutralization, centering, smoothness, and code-norm losses remain excluded. Architecture, 1000-step budget, data, candidates, and gates do not change. This is the terminal atom variant: success freezes it, failure ends the utility-memory paper path.

## D076 — Freeze the core atom and fit only one utility address

Date: 2026-08-25
Status: Accepted after EXP-015

EXP-015 passed every terminal atom gate on 225 OOF episodes/25 components. Current/base was 0.80194, oracle five-candidate utility +1.04799%, mean utility +0.52105%, harm 26.37%, and oracle-minus-mean CI `[+0.00439, +0.00618]`. The final 1000-step train refit is frozen by checkpoint hash.

No further atom, key, loss, optimizer, step size, reuse residual, or transport change is permitted. The next stage may fit only one source-entity-safe linear utility regressor whose score performs both retrieval and the semantic positive-utility decision. A separate fine router, risk head, calibrated threshold, or additional candidate network is prohibited.

## D077 — Register a top-1 zero-threshold unified address

Date: 2026-08-25
Status: Accepted before EXP-016 execution

EXP-016 uses one Ridge score over the exact-factorizable descriptor interaction `[c,s,c-s,c*s]`. The highest-scoring memory is the only retrieved candidate and is applied only when its predicted utility exceeds semantic zero. `K`, a learned threshold, PCA, fine router, and risk head are removed, leaving only `eta`, `alpha`, and capacity as visible inference hyperparameters.

Training/evaluation is leave-one-location-out with every held-location source entity removed from training. The causal candidate panel is sampled before target write and future/query frames are label-only. The fixed gates compare against matched-acceptance random and appearance addresses and must pass before any validation access.

## D078 — Add one adaptation-context scalar under unchanged address gates

Date: 2026-08-25
Status: Accepted after EXP-016 and before EXP-017

EXP-016 achieved +0.771% utility, 18.52% harm, positive association in every location, and significant superiority over appearance. Its random difference was +0.00216 but the component CI `[-0.00011, +0.00449]` crossed zero, so no artifact was accepted.

EXP-017 appends one already observable scalar: the normalized loss improvement produced by each context's own online TTT step. This directly represents adaptation dynamics while adding no loss, module, threshold, or hyperparameter. The pair score remains one exact factorized Ridge, now over 65-D states. All EXP-016 gates remain unchanged; failure ends address augmentation.

## D079 — Test deterministic geometry agreement before a learned fine router

Date: 2026-08-25
Status: Accepted after EXP-017 and before EXP-018

EXP-017 did not establish random superiority, so descriptor augmentation is stopped. Both EXP-016 and EXP-017 nevertheless show positive context-to-utility association, while the remaining gap is candidate-specific negative transfer after code transport.

EXP-018 restores the stronger visual-only coarse address, retrieves five candidates, and selects the one that most reduces the current 3D-track loss. Both coarse predicted utility and current agreement must exceed semantic zero. No fine model or threshold is learned. A positive component interval over both random and coarse top-1 is required; failure means a learned post-transport utility model is necessary and cannot be added without an explicit paper-scope decision.

## D080 — Test coarse fallback as the final parameter-free fine policy

Date: 2026-08-25
Status: Accepted after EXP-018 and before EXP-019

EXP-018 showed that positive current agreement isolates safe/useful cases—3.07% harm and significant random superiority—but strict abstention reduced utility to +0.418%. EXP-019 preserves those reroutes and falls back to coarse top-1 only when no positive-agreement candidate exists.

All zero thresholds, K=5, folds, comparators, and gates remain fixed. The fallback must significantly beat both random and coarse top-1. Failure ends heuristic routing and constitutes evidence that a compact learned post-transport utility model is necessary.

## D081 — Freeze the parameter-free fine policy for one-shot geometry validation

Date: 2026-08-25
Status: Accepted after EXP-019 and before EXP-020

EXP-019 passed every gate: +0.835% utility, 18.45% harm, 94.72% acceptance, random-difference CI `[+0.00041,+0.00512]`, and coarse-difference CI `[+0.00010,+0.00147]`. The final retrieval module is one 64-D utility-MIPS Ridge followed by current-geometry reranking of five candidates and coarse fallback. No learned fine router is required.

The atom/head, Ridge, eta, alpha, K=5, zero thresholds, reranking rule, reservoir policy, and capacity 64 are now frozen. EXP-020 is a one-shot validation of proxy and sparse-LiDAR geometry. Its outcome may accept or reject the paper model but cannot tune it; the closed EXP-009 test remains unavailable.

## D082 — Reject the frozen paper model and close EXP-020 validation

Date: 2026-08-25
Status: Accepted after the single locked EXP-020 validation

The EXP-015 atom plus EXP-019 address failed the registered broad-geometry
gate on 103 targets/17 components. Full memory significantly improved aligned
AbsRel over current-only TTT and self-supervised utility over matched random,
but current-only TTT itself worsened mean SILog and 3D EPE versus no TTT, no
primary LiDAR interval established full-memory superiority over random, and
proxy harm reached 33.01% against the locked 20% limit.

The compact utility-memory hypothesis is therefore not discarded, but this
frozen combination is rejected as a paper model. The failure localizes the
main problem to metric alignment and negative-transfer calibration after atom
meta-training: EXP-011 had already shown that the same one-loss online step can
be metric-healthy before the terminal proxy-oriented refit. EXP-020 validation
is now exposed and prohibited for model selection. No additional atom variant,
loss, threshold, or router may be tuned on EXP-009 test or EXP-020 validation.
A changed method requires a newly frozen component-disjoint benchmark and an
explicit paper-scope decision before training.

## D083 — Audit official-test revisits before choosing a replacement method

Date: 2026-08-25
Status: Accepted before EXP-021 metadata access

The local nuScenes `v1.0-test` metadata is the only untouched paper-scale data
currently available in the workspace. EXP-021 Stage 0 may read official pose,
timestamp, calibration, scene, location, and file-path metadata only. It may
check file existence but may not decode RGB/LiDAR or run a model. Physical
overlap connected components are the only admissible future split unit. Passing
the registered audit authorizes design of a new development/final-test split,
not model-output access. This preserves a clean endpoint before deciding between
a narrow self-supervised paper and one metric-aligned offline meta signal.

## D084 — Reserve all official-test revisit components for one terminal test

Date: 2026-08-25
Status: Accepted after EXP-021 Stage 0 and before Stage 1

The metadata audit passed with 107 edges, 96 overlap scenes, and 29 independent
components. One Boston component contains 54 edges, so a balanced two-way split
would either leave development with very few independent components or reduce
the terminal test below the desired target count. All 29 components are instead
reserved for one terminal test. Replacement-method design and selection must use
source-safe component OOF on the existing 25-component train benchmark only.
No official-test RGB, LiDAR, feature, or model output may be opened until one
method, all hyperparameters, all gates, and an artifact hash are frozen. Stage 1
may generate only the directional metadata manifest.

## D085 — Diagnose metric alignment before choosing the paper branch

Date: 2026-08-25
Status: Accepted before EXP-022 execution

EXP-020 alone cannot distinguish an intrinsically misaligned compact objective
from a final-refit generalization failure. EXP-022 therefore performs no model
fit or selection. On the existing train targets it decomposes foundation depth,
the metric-healthy EXP-011 current result, the EXP-015 zero-code readout, and the
EXP-015 one-step result. It also measures whether self-supervised future-loss
improvement predicts sparse-LiDAR improvement. The result determines which
single paper branch is scientifically justified; it may not authorize tuning on
EXP-020 or access to the locked EXP-021 terminal benchmark.

## D086 — Replace the offline proxy target, not the online loss or architecture

Date: 2026-08-25
Status: Accepted after EXP-022

EXP-022 localized the failure to the learned online update direction: zero code
exactly reproduces foundation depth, while the final one-step code improves
aligned AbsRel but worsens target-average SILog and 3D EPE. Track3D future gain
is significantly negatively associated with SILog and EPE gain. A narrow
proxy-only paper would not answer the central reviewer objection and is rejected
as the primary CVPR path.

The online method remains unchanged: one track3D loss, one local-code step, and
no extra module. The only admissible next signal is one offline sparse-LiDAR
scale-aligned log-depth loss evaluated on disjoint query frames. EXP-023 first
uses it only as an oracle selection label over frozen candidates. A new atom or
address fit is authorized only if this single label selects candidates that
improve SILog, aligned AbsRel, and 3D EPE together and provides component-level
headroom over the existing proxy oracle. No weighted multi-metric objective,
risk head, threshold sweep, or EXP-021 access is authorized.

## D087 — Register one scale-aligned log-depth utility label

Date: 2026-08-25
Status: Accepted before EXP-023 execution

The only EXP-023 label is the per-view median-aligned mean absolute log-depth
residual on sparse query LiDAR, averaged across query views. It has no metric
weights or learned parameters and is used only for offline oracle selection.
The frozen candidate set, atom, online step, transport, and residual do not
change. The label is viable only if its oracle improves the means of SILog,
aligned AbsRel, and 3D EPE together, obtains at least one positive component
interval over current, is no worse than the proxy oracle on all three means, and
beats the proxy oracle on its own risk with a positive component interval.
Failure ends this scalar label rather than adding weighted metric terms.

## D088 — Admit the single metric label for one atom feasibility fit

Date: 2026-08-25
Status: Accepted after EXP-023

The median-aligned absolute log-depth oracle passed every gate on 225 episodes
and 25 components. Relative to current TTT it improved SILog, aligned AbsRel,
and 3D EPE with positive component intervals for all three, and it beat the
track3D proxy oracle significantly on metric risk, SILog, and EPE. Because 68%
of individual frozen candidates were harmful, unconditional reuse is rejected.

EXP-024 may train a fresh frozen-key 8-D atom from scratch using only the equal
mean of current and best-candidate evaluations of this same metric loss. This
replaces all three EXP-015 proxy terms; it does not add a fourth term. Online
adaptation remains one track3D step at eta 0.0125 and reuse remains a 0.10 visual
residual. Training budget 1000, five component folds, optimizer settings, and
candidate construction remain fixed. No address, threshold, risk head, or
terminal-test access is authorized.

## D089 — Register the one-loss metric-aligned atom gate

Date: 2026-08-25
Status: Accepted before EXP-024 execution

EXP-024 trains from scratch and changes no architecture or online computation.
Its only outer objective is the equal mean of current and best-candidate
evaluations of the single EXP-023 metric loss. The fixed equal mean introduces
no tunable loss weight. Five component folds, 1000 steps, the existing optimizer,
five candidates, eta 0.0125, and residual 0.10 are immutable. Passing requires
current TTT and oracle reuse to improve all three primary metric means, at least
one positive component interval for each stage, and positive oracle headroom
over uniform candidates. Failure ends the from-scratch one-loss atom instead of
adding proxy preservation, regularizers, or weighted metric terms.

## D090 — Permit one terminal two-residual geometry objective

Date: 2026-08-25
Status: Supersedes only the post-failure next step in D089

EXP-024 ended the one-loss branch as registered and produced no checkpoint.
It nevertheless isolated a single missing property: the scale-aligned log
residual significantly improved SILog and 3D EPE and learned strong oracle
reuse, while aligned AbsRel alone worsened. The terminal EXP-025 objective is
therefore the fixed equal average of (a) the existing aligned absolute log-depth
residual and (b) aligned absolute relative-depth residual. The second residual
directly equals the failed evaluation quantity at sparse cells; the first
controls multiplicative/scale-invariant error. Equal averaging is immutable and
introduces no loss-weight hyperparameter.

This is a new explicitly registered two-residual branch, not a repair of the
failed EXP-024 record. Online TTT still has one track3D loss and inference adds
nothing. All other architecture, folds, candidates, optimizer, 1000-step budget,
eta, and residual strength remain fixed. EXP-025 is terminal: failure stops atom
objective development for this paper, and no third metric term, proxy term,
regularizer, or threshold tuning is permitted.

## D091 — Register EXP-025 as the terminal atom variant

Date: 2026-08-25
Status: Accepted before EXP-025 execution

The two sparse residuals and both fixed 0.5 coefficients are now immutable.
EXP-025 reuses the exact EXP-024 code path and gates, changing only the scalar
offline geometry loss. It may create a final refit checkpoint only after all
component-OOF gates pass. No EXP-020 or EXP-021 sensor/model output is accessed.
If any gate fails, the utility-addressed atom paper path is stopped rather than
trying another loss, seed, training budget, initialization, or module.

## D092 — Stop scalar atom-objective development

Date: 2026-08-25
Status: Accepted after EXP-025

EXP-025 failed the terminal gate and produced no checkpoint. Its current update
significantly improved aligned AbsRel but significantly worsened SILog and 3D
EPE, the inverse of EXP-024's trade-off. Both heads retained significant oracle
memory headroom across all three metrics. The unresolved issue is therefore not
whether useful past corrections exist, but whether one scalarized meta-objective
can learn a current plasticity direction that is Pareto-healthy.

No further loss, coefficient, seed, budget, initialization, or head variant is
authorized within the current compact-paper scope. EXP-020 remains exposed and
EXP-021 remains unopened. Further work requires an explicit scope decision:
either narrow the endpoint to aligned relative depth, abandon this paper path,
or make constrained/Pareto multi-objective plasticity a new central method. The
last option changes the paper's methodological contribution and cannot be
treated as another ablation.

## D093 — Approve the constrained multi-objective plasticity branch

Date: 2026-08-25
Status: Accepted by explicit user approval

The paper may continue with constrained/Pareto multi-objective plasticity as a
central training mechanism. This is a scope change, not another scalar-loss
variant. The method must remain compact: the inference architecture, one
online `track3D` loss, one local-code step, 8-D code, visual transport, and
fixed reuse residual do not change. The two sparse geometry objectives exist
only during offline meta-training, and no loss coefficient, extra head,
learned threshold, second online step, or terminal-data tuning is authorized.

Because gradient-consensus TTA and multi-objective gradient manipulation are
established prior art, the paper must not claim either generic idea as novel.
The defensible contribution remains transported, utility-addressed spatial
adaptation experience; constrained training is the geometry-health mechanism
needed to make that contribution valid on broad endpoints.

## D094 — Require a no-fit common-descent diagnosis before EXP-027

Date: 2026-08-25
Status: Accepted before EXP-026 execution

EXP-026 performs zero parameter updates on the existing train benchmark. At a
fresh head and two historical frozen heads it measures the gradient geometry
between aligned absolute log-depth and aligned absolute relative-depth outer
objectives. It compares raw equal averaging, exact raw two-objective MGDA, and
a parameter-free unit-normalized bisector.

The gate is registered in the EXP-026 record. A new fit is authorized only if
metric conflict and raw-average sacrifice are material, while the normalized
bisector supplies non-degenerate common descent across all anchors. This
prevents adopting a fashionable optimizer without evidence that it addresses
the repository's actual failure. EXP-021 remains unopened.

## D095 — Authorize one coefficient-free Pareto atom fit

Date: 2026-08-25
Status: Accepted after EXP-026

EXP-026 passed all five registered gates. At the learned EXP-006 and EXP-015
anchors, component-balanced gradient conflict was 35.43% and 27.30%, while raw
equal averaging sacrificed at least one endpoint in 32.83% and 22.89%. The
unit-normalized bisector was strict common descent for all 675 measured
anchor/episode pairs and was far from antiparallel degeneracy.

EXP-027 may therefore train one fresh head with two separately differentiated
offline objectives and replace their scalar sum by the arithmetic mean of
their unit gradients. It retains the exact EXP-024/025 architecture, folds,
1000-step budget, AdamW implementation, online loss, code step, candidate set,
and reuse residual. No loss coefficient, task-weight solver, CAGrad radius,
risk head, threshold, or new module is introduced.

The first-order common-descent statement applies to the synthesized gradient,
not automatically to AdamW's preconditioned parameter displacement. EXP-027
must log the realized per-step displacement alignment and pass the same broad
geometry OOF gates before any checkpoint or address fit is accepted.

## D096 — Reject EXP-027 and localize the remaining optimizer failure

Date: 2026-08-25
Status: Accepted after EXP-027

EXP-027 created no checkpoint. It improved SILog and 3D EPE, reduced the
EXP-024 AbsRel damage by more than half, and retained significant three-metric
oracle reuse headroom. It nevertheless worsened mean AbsRel by 0.00279 and
realized AdamW common descent on only 72.54% of component-balanced steps.

No objective, loss weight, head, seed, budget, or inference module may change.
The remaining causal variable is the offline optimizer displacement: AdamW may
rotate a valid synthesized common gradient outside the feasible descent cone.

## D097 — Register one parameter-free feasible-displacement safeguard

Date: 2026-08-25
Status: Accepted before EXP-028 execution

EXP-028 is a same-seed paired optimizer ablation. It preserves an AdamW proposal
when that proposal is common descent. Otherwise it replaces the direction by
the already registered unit-normalized bisector and preserves the proposal's
L2 norm. This adds no tunable margin, coefficient, solver, line search, loss,
module, or online computation.

All broad-geometry and reuse gates remain unchanged, and realized common
descent must be 100%. Failure rejects this optimizer candidate and creates no
checkpoint. EXP-021 remains unopened.

## D098 — Freeze the EXP-028 atom and require metric-utility addressing

Date: 2026-08-25
Status: Accepted after EXP-028

EXP-028 passed every registered OOF gate and produced checkpoint SHA-256
`3ebf194f3a28876014e46d1d3bbdbcd1422cfb8ebdba48f3d16635520ca787ae`.
Current TTT improved all three primary means, with positive SILog and EPE
component intervals. Log-risk oracle reuse further improved all three with
positive intervals. The safeguard was needed on 29.28% of OOF steps and
preserved realized common descent on 100%.

The atom architecture, checkpoint, online step, transport, and residual are
now frozen. Because 39.91% of raw candidates remain harmful, unconditional or
appearance-only reuse is not authorized. The next learned object may only be
one factorized linear address trained on the existing single aligned-log future
utility label, with semantic zero acceptance and source-location exclusion.

## D099 — Register one metric-utility linear address

Date: 2026-08-25
Status: Accepted before EXP-029 execution

EXP-029 freezes the EXP-028 checkpoint and changes the address target from the
rejected self-supervised proxy to the single EXP-023 aligned-log metric utility.
The model remains one exact-factorizable Ridge score with semantic zero
acceptance. Ridge alpha 1, panel 64, source-excluded location folds, top-1, and
all causal/query boundaries are fixed implementation choices inherited from
EXP-016.

Passing requires positive association in every location, at most 30% selected
harm, and component-level superiority over matched random plus higher utility
than appearance. No fine router, risk classifier, threshold sweep, or EXP-021
access is authorized.

## D100 — Freeze the metric address pending an absolute-geometry audit

Date: 2026-08-25
Status: Accepted after EXP-029

EXP-029 passed every gate on 13,631 causal pairs. The unified score achieved
positive association in all four held locations, +0.00320 component-balanced
metric utility, 11.44% harm, and significant superiority over both matched
random and appearance. Artifact SHA-256 is
`d8b81fff36d5cb5635c194a63b422edf700c0683b7f7eb2d477be67091430984`.

The address, zero acceptance rule, atom checkpoint, and all inference settings
are frozen. EXP-030 must recompute OOF selected candidates and verify SILog,
aligned AbsRel, and 3D EPE directly. This is a no-fit audit, not another model
selection stage. EXP-021 may be opened only if the frozen full system passes.

## D101 — Register the frozen full-system geometry gate

Date: 2026-08-25
Status: Accepted before EXP-030 execution

EXP-030 performs no fit and recomputes the frozen causal panels using the
EXP-028 checkpoint and source-safe OOF EXP-029 predictions. Top-1/zero,
candidate panels, descriptors, transport, and residual are immutable. Matched
random and appearance policies use identical acceptance.

The full policy must improve mean SILog, aligned AbsRel, and 3D EPE over each
of current-only, random, and appearance, with at least one positive component
interval per comparison family. Failure keeps EXP-021 locked; success alone
authorizes assembly of the terminal artifact.

## D102 — Freeze the complete development system for terminal evaluation

Date: 2026-08-25
Status: Accepted after EXP-030

EXP-030 passed every no-fit gate on 217 targets and 25 components. The full
policy improved SILog, aligned AbsRel, and 3D EPE over current-only, matched
random, and appearance; all nine component intervals were positive.

The terminal candidate is now immutable: EXP-028 checkpoint, EXP-029 address,
one `track3D` step at 0.0125, 8-D code, visual transport, residual 0.10, top-1
semantic-zero reuse, causal write-after-predict, and the registered bounded
memory protocol. EXP-031 may open EXP-021 exactly once. No development result
after this decision may change the model, threshold, comparator, or gate.

## D103 — Register the one-shot EXP-031 terminal protocol

Date: 2026-08-25
Status: Accepted before any EXP-021 sensor/model access

EXP-031 uses all 214 locked directional episodes and three official-location
streams. It converts only the 96 selected scene metadata, creates one frozen
RGB/geometry/tracker cache with train-fitted PCA, then evaluates the immutable
EXP-028/029 candidate. The online bank is deterministic reservoir-64 per
location, writes after prediction, retrieves top-1, and accepts only above
semantic zero.

The registered controls are current-only, same-bank random expectation, and
appearance retrieval. Full memory must improve all three primary means over
each control with at least one positive component interval per comparison.
Terminal outputs cannot authorize a repair, new seed, threshold, capacity,
loss, or module.

## D104 — Preserve the EXP-031 failure and separate coverage accounting

Date: 2026-08-25
Status: Accepted after the one terminal evaluation

EXP-031 improved every registered primary mean over current-only, same-bank
random, and appearance. All comparison-family rules passed, but the complete
registered gate failed because 187 evaluated targets were below the frozen 190
minimum. The terminal result remains immutable and may not be relabeled a pass.

EXP-032 audited only manifest metadata and the written result. The 214
directional episodes collapse to 188 unique targets; one target is the first
event in its location and therefore has an empty causal bank. The evaluator
included all 187 causally eligible targets, zero eligible targets failed metric
validity, and all 29 components were retained. Thus the coverage criterion was
an infeasible unit-accounting error rather than model or data attrition. Paper
claims may describe the geometry evidence as qualified positive terminal
evidence only if the registered coverage failure is disclosed alongside it.

## D105 — Close model selection and audit paper efficiency next

Date: 2026-08-25
Status: Accepted after EXP-032

The EXP-028 atom, EXP-029 address, and complete EXP-031 inference policy remain
the selected frozen paper model. Neither the terminal comparisons nor the
coverage audit authorizes another nuScenes variant, seed, threshold, loss,
module, or terminal run.

The next experiment is a no-fit efficiency/complexity audit of this exact
candidate. It may measure latency, peak memory, persistent bank size, and
search scaling, but cannot alter the implementation to improve those numbers.
After that audit, the remaining model-evidence expansion must be an independent
dataset/backbone transfer, not additional selection on exposed nuScenes data.

## D106 — Accept the efficiency audit and expose storage as the main cost

Date: 2026-08-25
Status: Accepted after EXP-033

The frozen method adds approximately 1.996 ms after 292.328 ms of separate
FastVGGT feature/geometry and tracker passes on an A100 (eight 224×224 views).
The exact bank-64 address takes 0.002 ms CPU, and learned additions contain
288,386 parameters. Runtime overhead is therefore small in this implementation.

The actual float32 plasticity record is 0.602 MiB and reservoir-64 tensor
payload is 38.52 MiB, excluding Python containers. Per-token visual keys account
for 83.1% of each record. The paper must report this storage rather than imply
that bounded memory is free. EXP-033 does not authorize key compression or
another capacity selection. The next evidence expansion is an independent
dataset/backbone feasibility audit.

## D107 — Correct only the EXP-034 no-access gate polarity

Date: 2026-08-25
Status: Accepted after the metadata-only v1.0 execution

EXP-034 v1.0 found sufficient TUM metadata but failed mechanically because the
required false states `sensor_decoded` and `model_output_accessed` were inserted
as false values into `all(checks)`. The v1.0 result remains preserved. Version
1.1 changes only those predicates to `no_sensor_decoded=true` and
`no_model_output_accessed=true`, with new output paths. No association,
threshold, sequence, count, sensor data, or model output changes.

## D108 — Register one frozen TUM transfer stress test

Date: 2026-08-25
Status: Accepted after EXP-034 v1.1

EXP-034 passed with 223 contexts and 111 causal revisit targets, but only three
sequences and a 4/98/9 target split. EXP-035 is therefore descriptive
cross-domain evidence rather than a replacement independent benchmark.

The complete nuScenes-selected model and policy are immutable. TUM may supply
RGB online and dense query depth offline; no TUM observation may fit the head,
address, PCA, threshold, loss, capacity, or checkpoint. Full memory must improve
all three sequence-balanced primary means over current-only, same-bank random,
and appearance. Failure is terminal for zero-shot TUM and cannot authorize a
repair.

## D109 — Accept descriptive dataset transfer and freeze the final model

Date: 2026-08-25
Status: Accepted after EXP-035

Without any TUM fitting, the immutable full policy improved sequence-balanced
SILog, aligned AbsRel, and 3D EPE over current-only, same-bank random, and
appearance. It improved all three metrics over current-only within each of the
three sequences. This supplies descriptive cross-dataset support.

The address accepted 100% of targets, and the four-target Freiburg1-desk
sequence favored random/appearance controls. With only three imbalanced groups,
EXP-035 does not establish cross-domain risk calibration or paper-level
inference. No TUM repair is authorized. The final architecture and all model
values are now closed; remaining work is paper evaluation infrastructure,
external baselines, fixed qualitative outputs, and writing—not another method
variant.

## D110 — Register matched causal CUT3R/TTT3R baselines

Date: 2026-08-25
Status: Accepted before external-model evaluation

The locally available TTT3R repository and official CUT3R final checkpoint can
evaluate both recurrent update rules. EXP-036 feeds the exact EXP-035 event
order and marks query frames `update=false`, preserving the causal future
boundary. The official 512 preprocessing, no crop, sequence-only reset, and all
metrics are frozen before execution.

This is an absolute external-method comparison, not a controlled adaptation
ablation: architectures, training data, resolution, and state differ. It cannot
select or repair Revisit3D. tttLRM is retained as a conceptual closest method
because its released calibrated-camera Gaussian renderer does not satisfy the
same query-read-only pointmap interface without a different protocol.

## D111 — Treat absolute SOTA competitiveness as a paper blocker

Date: 2026-08-25
Status: Accepted after EXP-036

EXP-036 evaluated both official modes on all 111 matched causal TUM targets.
TTT3R achieved SILog 15.727, aligned AbsRel 0.0781, and 3D EPE 0.2246 m;
Revisit3D full achieved 28.462, 0.2301, and 0.4589 m. CUT3R was also far better
than the custom head. Official baselines use 512 inputs and different training,
so this is not a controlled mechanism ablation, but the absolute gap blocks a
broad top-tier reconstruction-framework claim.

The frozen Revisit3D result remains valid evidence that utility-addressed local
adaptation improves its own backbone. It is not evidence of competitive
reconstruction quality. No post-result change on exposed nuScenes/TUM is
authorized under the closed model. A new CUT3R/TTT3R-class backbone integration
would be a deliberate new research branch with new development and held-out
data, not a repair or ablation of the final candidate.

## D112 — Archive the FastVGGT mechanism candidate and open a competitive-carrier branch

Date: 2026-08-25
Status: Accepted by explicit project decision

The complete EXP-036 state is preserved as an immutable mechanism-proof
baseline with an archival Git ref, a standalone Git bundle, artifact copies,
and SHA-256 verification. Its code, checkpoints, exposed nuScenes/TUM results,
and interpretations may not be silently changed by the new branch.

The paper-first project now reopens one material scope variable: the absolute
geometry carrier. The local plasticity code, explicit transport, and
future-utility address remain the method hypothesis. The first new experiment
must be a no-fit carrier diagnostic, not another memory module: determine
whether the official FastVGGT geometry head supplies sufficient absolute
quality while preserving the existing token interface, or whether integration
must move to a CUT3R/TTT3R-class recurrent state. Exposed TUM data may be used
only for this declared engineering diagnosis and may not become final
held-out evidence.

## D113 — Reject the minimal official FastVGGT carrier and select a recurrent carrier

Date: 2026-08-25
Status: Accepted after EXP-037

The official frozen FastVGGT head improved every primary error over the custom
head by 35--51%, confirming that the archived framework's main absolute-quality
bottleneck was its decoder. It nevertheless reached 1.431 times TTT3R aligned
AbsRel and 1.322 times TTT3R 3D EPE, failing the pre-registered 1.25 maximum.

The next implementation must therefore use a CUT3R/TTT3R-class recurrent
geometry carrier. To keep one-paper scope compact, it may add only one local
plasticity residual interface and reuse one utility address. It must not add a
second router, risk network, auxiliary reconstruction head, or collection of
online losses. TTT3R remains an external baseline: the proposed method must
store and retrieve transported local adaptation experience rather than merely
rename TTT3R's recurrent scene state.

The exposed TUM diagnostic is now closed for selection. Carrier integration
must establish its development and held-out evidence on new source-safe
partitions before any paper claim.

## D114 — Correct only the EXP-038 attention and distance implementations

Date: 2026-08-25
Status: Accepted after EXP-038 v1.0 implementation audit

EXP-038 v1.0 is preserved. Its native parity comparison used SDPA in the custom
step while official lighter inference explicitly materialized attention, and
its `torch.cdist` computation produced false nonzero diagonal distances for
large canonical coordinates. These are implementation mismatches, not evidence
against zero-residual parity or 3D transport.

Version 1.1 changes only `return_attn=true` and direct-difference Euclidean
distance. It retains the same frames, checkpoint, code dimension, basis seed,
diagnostic step, and gates. No result-dependent hyperparameter or architectural
change is authorized.

## D115 — Accept the CUT3R local-code interface and freeze its conceptual scope

Date: 2026-08-25
Status: Accepted after EXP-038 v1.1

The corrected audit obtained exact native and zero-code parity, a finite
one-loss descent direction, nonzero geometry response, exact identity
transport, and finite adjacent-frame 3D transport. The v2 interface is accepted
for premise testing.

Its paper scope is frozen conceptually: official frozen CUT3R recurrence and
geometry head; one 8-D code per decoder patch; one shared linear residual basis;
one symmetric predicted-3D online loss; explicit nearest-neighbor transport in
the predicted canonical frame; and one future-geometry utility address. The
next evidence may train the one basis and later fit the one address, but may not
add another decoder, recurrent state, loss family, learned transport network,
risk head, or threshold collection.

EXP-038 establishes interface feasibility only. Before any fitting, a new
source-safe development/held-out benchmark must be registered. Exposed TUM and
previous nuScenes partitions cannot select the integrated model.

## D116 — Correct EXP-039 by excluding unavailable RGB pairs before subsampling

Date: 2026-08-25
Status: Accepted after EXP-039 v1.0 metadata audit

EXP-039 v1.0 is preserved. It passed all scene, pair, split, disjointness, and
no-access checks but referenced one absent local RGB file in one terminal pair.
Version 1.1 removes unavailable four-frame pairs before the already registered
deterministic per-scene subsampling. This is a data-integrity correction only;
all revisit thresholds, minimum/maximum pair counts, scene assignment, split
sizes, and gates remain unchanged. No pixel was decoded and no model output was
accessed.

## D117 — Accept the DL3DV split and require a train-only oracle premise next

Date: 2026-08-25
Status: Accepted after EXP-039 v1.1

The corrected audit locked 63/14/14 scene-disjoint train, validation, and
terminal roles with 982/213/224 pairs. The terminal manifest hash is fixed
before pixel/model access and may not be opened during model selection.

The next experiment may open only 32 deterministically selected train pairs to
test whether the fixed 8-D coordinate supplies (a) beneficial current TTT and
(b) oracle past-code utility after predicted-3D transport. This premise must be
tested before training the shared basis or fitting an address. Validation and
terminal images remain closed.

## D118 — Reject raw canonical-3D update reuse on the CUT3R carrier

Date: 2026-08-25
Status: Accepted after EXP-040

The frozen 8-D coordinate produced positive current TTT at source and target,
but adding the correctly paired 3D-transported source code worsened target loss,
lost to a spatial shuffle, and harmed 68.75% of pairs. A physical revisit plus
nearest predicted 3D correspondence is therefore insufficient evidence of
adaptation compatibility on this carrier.

No address, bank, risk gate, or basis meta-training is authorized from this
result alone. One train-only decomposition may compare untransported,
visual-feature, canonical-3D, and shuffled code placement and measure alignment
with the target current direction. This adds no module and does not open
validation or terminal data. If no carrier produces positive oracle utility,
the v2 memory hypothesis must be narrowed or abandoned rather than hidden by a
learned router.

## D119 — Replace raw-gradient reuse with one learned plasticity coordinate

Date: 2026-08-25
Status: Accepted after EXP-041

Untransported, frozen-feature visual, and predicted-3D carriers all produced
negative mean agreement with the target current code and negative future gain.
The failure is therefore update-coordinate incompatibility, not only 3D
correspondence. The original broad claim that past raw TTT directions are
naturally reusable is rejected on the competitive carrier.

One compact revision remains within the paper scope: offline meta-training of
the already implemented 6,144-parameter shared `8 -> 768` basis, using the same
symmetric future consistency loss evaluated after current and revisit code
application. Online inference remains one loss, one normalized step, one local
code, and one transport. No alignment loss, second decoder, hypernetwork,
risk head, or address is added.

The first learned-basis experiment must split EXP-039 train scenes internally,
fit on one subset, and evaluate oracle visual reuse on disjoint train scenes.
Validation remains closed. Failure ends this compact v2 memory direction;
success alone authorizes one frozen validation run and only then address fitting.

## D120 — Stop compact competitive-carrier v2 after the learned-coordinate gate

Date: 2026-08-25
Status: Accepted after EXP-042

The single registered fit completed exactly 128 steps on 32 train scenes and
was audited once on 16 disjoint train scenes. It improved current TTT in every
audit scene and produced a small positive oracle-reuse point estimate, but
failed the required positive source/target code-agreement check. Reuse and
shuffle advantages also had scene-bootstrap intervals spanning zero, and
46.88% of pairs were harmed.

This is evidence that the small basis can learn a stronger current adaptation
coordinate, not evidence that it stores reusable adaptation experience. The
checkpoint is retained as a rejected artifact and must not be promoted to
validation or used to fit an address. No post-result learning-rate, epoch,
threshold, transport, or loss change is allowed under EXP-042.

D119's compact v2 allowance is exhausted. A next experiment requires an
explicit project-level scope decision because it must either remove continual
reuse from the primary claim or introduce a materially different plasticity
objective/representation. The archived FastVGGT mechanism candidate remains
unchanged but cannot satisfy the absolute competitive-carrier requirement.

## D121 — Reopen only the plasticity objective with exact future-utility differentiation

Date: 2026-08-25
Status: Accepted by explicit project decision

The project approves one material revision while preserving the one-paper
architecture budget. The carrier, local-code capacity, online loss, online
step, transport, and head interface remain unchanged. Offline training may now
differentiate future geometry consistency through the source and target code
generation steps so the basis controls both the update coordinate and its
future effect.

This is a replacement for EXP-042's detached first-order training, not an added
module or auxiliary loss. The deterministic optimizer and one-pass budget stay
the same. Previously exposed train scenes may be used for fitting, but the
remaining 15 unopened EXP-039 train scenes must be locked as a new internal
audit before any pixel access. Validation and terminal remain closed.

Functional future-utility confidence intervals replace code cosine as the
primary gate because basis coordinates are only meaningful through their
decoded effect. Failure stops the exact-meta realization. Success permits one
frozen validation run; it does not automatically authorize an address or bank.

## D122 — Reject unconditional exact-meta reuse and isolate zero-agreement routing

Date: 2026-08-25
Status: Accepted after EXP-043

EXP-043 passes current adaptation but fails both registered memory-specific
confidence bounds. Exact meta-training alone is not sufficient evidence for
unconditional reuse, and the checkpoint may not proceed directly to
validation as the registered EXP-043 method.

The failure is heterogeneous rather than uniformly null. On the exposed audit,
oracle fallback has substantial headroom and current/memory code cosine has
`0.752` correlation with future utility. A post-hoc positive-sign policy reduces
harm from 50.00% to 1.67% and has a positive development interval. Because this
was inspected after EXP-043, it is only a design lead.

One parameter-free routing revision is authorized: apply the transported code
iff its mean cosine with the current code is strictly positive. The value zero
is algebraic descent agreement, not a calibrated threshold. EXP-044 must record
the post-hoc development calculation transparently. Only after its code and
policy are frozen may a new EXP ID open validation once. No learned address,
risk head, threshold sweep, new loss, or memory bank is authorized yet.

## D123 — Freeze zero-agreement routing for one-shot validation

Date: 2026-08-25
Status: Accepted after post-hoc EXP-044

EXP-044 transparently confirms the inspected development pattern: the strict
positive-agreement rule retains nearly all oracle fallback gain and reduces
harm to one of 60 pairs. This remains post-hoc and cannot support the paper
claim by itself.

The EXP-043 checkpoint, one-step loss, visual transport, and threshold zero are
now immutable. EXP-045 may open every EXP-039 validation pair once and must
compare unconditional reuse plus an independently scored spatial-shuffle
control. No validation result may tune the threshold, basis, step size,
transport, or gate. Terminal remains closed.

## D124 — Accept zero-agreement routing and require causal retrieval next

Date: 2026-08-26
Status: Accepted after EXP-045

The immutable zero-agreement candidate passed all validation gates on every
scene. It significantly beats current-only, unconditional reuse, and a spatial
shuffle routed at the same acceptance fraction. This establishes a compact
utility decision for a supplied memory candidate without learned routing.

The method is not yet deployable because EXP-045 uses the manifest's physical
revisit candidate. The next experiment must replace pair identity with a causal
bank of earlier source records and compare parameter-free candidate selection:
frozen appearance similarity, maximum current/memory descent agreement, and a
matched random record. The exact-meta basis and positive-sign application rule
remain frozen. Validation may now serve only as development for this new bank
protocol; terminal remains unopened until selection and capacity are fixed.

## D125 — Accept agreement addressing but require every-frame bounded streaming

Date: 2026-08-26
Status: Accepted after EXP-046

Maximum code agreement beats pooled appearance and matched random addressing
with positive intervals in every development scene. Its selected record matches
the manifest pair only 17.37%, showing that physical pair identity is not the
utility target. The parameter-free agreement address is accepted for further
development.

EXP-046's bank writes are nevertheless curated by the pose-built manifest. No
terminal claim is authorized from that protocol. The next experiment must
process the ordinary RGB stream continuously, write every eligible frame after
prediction, and use a deterministic capacity-16 reservoir. FIFO-16,
same-reservoir appearance, and same-reservoir random are fixed controls. Only
this experiment may select retention before terminal is opened.

## D126 — Select FIFO-16 provisionally and require an optimization-equivalence audit

Date: 2026-08-26
Status: Accepted after EXP-047

Every-frame reservoir agreement remains useful and beats appearance/random, but
FIFO agreement is significantly better in every scene and has zero observed
harm. Reservoir superiority and a strong long-term-retention interpretation are
rejected on this development stream. FIFO-16 is the provisional retention rule.

The selected FIFO records are only 6.18 frames old on average. This creates a
central novelty risk: transported recent memory may approximate a second
current TTT step. EXP-048 must compare those alternatives at equal normalized
step size on the exact full-stream protocol. No terminal access or memory claim
is authorized until FIFO memory beats that control with a positive confidence
bound.
