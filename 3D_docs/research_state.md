# Current Research State

Last updated: 2026-08-25

## Research goal

Develop a streaming 3D/4D reconstruction framework that learns compact local corrections from current geometric evidence and reuses useful adaptation experience without uncontrolled interference or memory growth.

The first milestone targets depth/point geometry under static revisits. Pose adaptation, dynamic point tracking, and 4D memory remain later milestones.

## Current central claim

> **A spatial local TTT update is reusable after visual transport, and a learned utility model conditioned on appearance and observable adaptation history can choose useful past updates better than place identity, appearance similarity, current loss, random retrieval, or uniform memory averaging.**

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

## Experiment status

- EXP-001 — tttLRM fast-weight premise probes: completed, negative.
- EXP-002 — independent Revisit3D benchmark and objective health: completed.
- EXP-003 — compact/global/slot reuse: completed, negative for selectivity.
- EXP-004 — retrieval keys and vector routing: completed, partial/negative.
- EXP-005 — dense oracle 3D transport: completed as controlled evidence; test split closed.
- EXP-006 — expanded train OOF architecture/model selection complete; one-shot validation model locked in D023; validation not yet accessed.

## Current hypothesis status

- H1 local reusable adaptation: supported on expanded train OOF.
- H2-P predicted geometry carrier: rejected.
- H2-E predicted geometry primary router evidence: rejected.
- H3 online utility observability: supported on expanded train OOF.
- H4-U learned utility routing: supported on expanded train OOF; validation pending.
- H4-R explicit risk classifier: rejected in current form.
- H5 continual consolidation: gated on one-shot validation.
- H6 dynamic 4D extension: open.

## Next step

Run exactly one validation evaluation using the D023-locked model and unchanged 14 validation episodes. Do not tune on that result. If utility routing reproduces with acceptable harm, close EXP-006 and begin EXP-007 continual memory/consolidation. If it fails, report the failure and revisit the scientific hypothesis using train-only/new data rather than adapting to validation.
