# Current Research State

Last updated: 2026-08-25

## Research goal

Develop a streaming 3D/4D reconstruction framework that learns compact local corrections from current geometric evidence and reuses useful adaptation experience without uncontrolled interference or memory growth.

The first milestone targets depth/point geometry under static revisits. Pose adaptation, dynamic point tracking, and 4D memory remain later milestones.

## Current central claim

> **A spatial local TTT update is reusable after visual transport; a learned observable-utility model can rank candidate updates; and a distinct frozen token-set address plus causal utility history can bound a continual atom bank while retaining most reuse benefit.**

Predicted 3D alignment is neither the fast-code carrier nor a primary router input. Correct memory is defined by causal future utility, not paired episode identity.

## Locked EXP-006 architecture

1. Frozen VGGT supplies dense tokens and controlled tracking evidence.
2. A custom plasticity head performs exactly one current-context TTT step on an 8-D per-token code.
3. Past codes are transported to current tokens by visual correspondence.
4. Current/source descriptors and observable adaptation-history/visual-transport statistics enter a regularized utility router.
5. The router selects one candidate only when predicted utility is positive; otherwise it returns current-only TTT.
6. Accepted memory is added after current TTT as a fixed 0.10 bounded residual.
7. Predicted Sim(3) evidence and neural risk classification are ablations, not primary-path components.
8. A second TTT step is prohibited in the primary path because it was unstable.

The locked Stage-2 router is `StandardScaler → PCA(16) → Ridge(alpha=1)` over descriptor interactions plus 20 non-alignment adaptation-history scalars. This compact linear model generalized better than the tested neural risk head on current data.

## Provisional EXP-007 continual bank

1. The learned per-token key is used only for local code transport.
2. A separate frozen DINOv2 ViT-L/14 four-view token-set key predicts consolidation compatibility; EXP-009 train OOF selected it over VGGT reconstruction tokens.
3. The bank is capacity-bounded; capacity 8 is the current benchmark setting, not a universal architectural constant.
4. Past-only predicted utility history prioritizes records for retention.
5. Consolidation produces a small candidate set; the EXP-006 utility router ranks transported codes.
6. Bucket similarity does not accept/reject reuse. Safety still comes from the fixed 0.10 residual and current-only fallback.

The dual-address design was selected because the learned transport key failed as a consolidation key. EXP-009 then resolved the provisional backbone choice: on 225 positive/225 strict-negative fully unseen train pairs, DINOv2 achieved 0.936 leave-one-location-out AUC versus VGGT 0.744, with a minimum location AUC of 0.900. DINOv2 is now the selected consolidation representation, pending causal utility validation.

EXP-008 corrected the stream order: across 71 unique target contexts in true nuScenes capture time, the frozen-key bank reached +2.650% utility and 5.63% harm versus appearance diversity +2.387%/7.04%. The paired component interval was [+0.019%, +0.486%], and the observed utility exceeded 96.1% of 1,000 matched compression nulls (p=0.03996). Thus the dual-address bank is the selected static-revisit architecture, while its independent-scene generalization is still open.

## Authoritative expanded train evidence

All results are exact train-only OOF estimates over 76 directional episodes, 38 undirected overlaps, 19 physical-overlap components, and 380 candidates. Original validation/test episodes were protected by scene-disjoint component construction and remain unchanged.

- Current one-step TTT/base: **0.6622**.
- Global reuse: **+0.27%**; untransported local: **+0.45%**, harm **22.37%**.
- Visual local transport: **+1.80%**, benefit **63.68%**, harm **4.47%**, coverage **100%**.
- Predicted geometry: **+1.52%**, harm **5.20%**, coverage **70.79%**.
- Geometry+appearance: **+1.64%**, harm **8.55%**, coverage **70.79%**.
- Visual mean: **+1.84%**, benefit **75.0%**, harm **3.95%**.
- Oracle best visual candidate: **+3.31%**.
- Locked no-alignment utility router: **+2.80%**, benefit **82.89%**, harm **2.63%**, regret **0.51%**.
- Current-objective heuristic: **+1.38%**, harm **2.63%**.
- Appearance similarity: **+1.59%**, harm **7.89%**.
- Matched physical identity: **+1.47%**, harm **9.21%**.
- Router minus visual mean: **+0.98 percentage points**, component-bootstrap 95% CI **[+0.63, +1.45]**.
- Router minus random: **+1.02 points**, CI **[+0.66, +1.50]**.
- Router minus matched identity: **+1.35 points**, CI **[+0.89, +1.99]**.

Risk-label diversity now passes: 242 beneficial, 121 neutral, and 17 harmful candidates across three folds. However, neural risk heads did not improve selected harm. The explicit risk-classifier hypothesis is rejected in its current form.

## One-shot validation evidence

The D023/D024-locked model was evaluated exactly once on the unchanged 14-episode validation split (2 physical-overlap components):

- Locked utility router: **+1.785%** mean utility, **0%** deadband harm, **0.503%** regret, **100%** accept.
- Visual mean: **+1.407%**, 7.14% harm.
- Current objective: **+1.290%**, 0% harm.
- Appearance similarity: **+1.073%**, 14.29% harm.
- Matched identity: **+0.788%**, 14.29% harm.
- Random candidate expectation: **+1.402%**.
- Oracle candidate: **+2.288%**.
- Router minus visual mean: **+0.378 points**; two-component descriptive bootstrap CI **[+0.064, +0.430]**.

All five D024 descriptive checks passed. The router accepted every episode, so validation supports candidate ranking and bounded reuse but does not establish learned rejection. Its absolute two-component bootstrap CI crosses zero because one small component has neutral mean utility (−0.67%). This is a feasibility gate, not paper-level generalization.

## Experiment status

- EXP-001 — tttLRM fast-weight premise probes: completed, negative.
- EXP-002 — independent Revisit3D benchmark and objective health: completed.
- EXP-003 — compact/global/slot reuse: completed, negative for selectivity.
- EXP-004 — retrieval keys and vector routing: completed, partial/negative.
- EXP-005 — dense oracle 3D transport: completed as controlled evidence; test split closed.
- EXP-006 — completed; expanded train OOF and one-shot validation support the locked utility-routed visual-memory feasibility claim.
- EXP-007 — completed as train-only pseudo-stream feasibility; H5 partially supported and the dual-address bounded bank selected provisionally.
- EXP-008 — completed on train; true capture-time replay supports the dual-address bounded bank beyond a matched compression null.
- EXP-009 — active: unseen benchmark frozen; train-only DINOv2 consolidation screening passed; causal local-reuse and bank utility transfer are next.

## Current hypothesis status

- H1 local reusable adaptation: supported on expanded train OOF and descriptive validation.
- H2-P predicted geometry carrier: rejected.
- H2-E predicted geometry primary router evidence: rejected.
- H3 online utility observability: supported on expanded train OOF and descriptive validation.
- H4-U learned utility routing: supported for ranking on expanded train OOF and descriptive validation; reject behavior remains unproven.
- H4-R explicit risk classifier: rejected in current form.
- H5 continual consolidation: partially supported; real chronological and independent-benchmark generalization remain open.
- H6 dynamic 4D extension: open.

## Next step

On a fixed EXP-009 train-only geometry pilot, test whether the locked VGGT plasticity head and one-step TTT objective retain positive local visual-reuse utility on unseen scenes. Then replace only the bank consolidation address with the selected frozen DINOv2 key and test causal utility against VGGT, appearance-diversity, FIFO, oracle-scene, and matched-permutation controls. Do not access EXP-009 validation/test pixels until the new train router/bank is locked.
