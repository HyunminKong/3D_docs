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
