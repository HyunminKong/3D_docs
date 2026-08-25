# Current Research State

Last updated: 2026-08-25

## Research goal

Develop a streaming 3D/4D reconstruction framework that learns compact local corrections from current geometric evidence and reuses useful adaptation experience without uncontrolled interference or unbounded memory growth.

The completed first milestone targets depth/point geometry under static cross-traversal revisits. Pose adaptation, dynamic point tracking, and 4D memory remain later milestones.

## Supported central claim

> **A spatial local TTT correction is reusable after token-level visual transport, and its future utility can be addressed from current/source observable geometry descriptors. A fixed utility router can then apply useful memories through a bounded residual, while a capacity-64 causal reservoir retains the benefit on fully unseen revisit components.**

This is an adaptation-utility claim, not a place-recognition claim. Generic DINOv2 place compatibility, predicted Sim(3) transport, and raw update similarity are not the selected memory address.

## Final static-revisit architecture after EXP-009

1. A frozen VGGT/FastVGGT foundation supplies dense tokens, predicted geometry, and controlled frozen-track evidence.
2. A custom plasticity head performs exactly one current-context TTT step on an 8-D per-token fast code. Foundation and head weights remain frozen online.
3. Each causal memory record stores the local code/key, a pooled 64-D transport descriptor, and observable adaptation statistics.
4. A train-only Ridge pair scorer over `[current, source, current-source, current*source]` descriptors is compiled exactly into a 64-D maximum-inner-product address.
5. Each official-location stream uses deterministic reservoir retention with capacity 64 and retrieves K=5 candidates.
6. A source code is moved to current tokens by learned visual correspondence and added after current TTT as a fixed 0.10 residual.
7. A frozen `StandardScaler → PCA(16) → Ridge(alpha=1)` utility router selects one candidate only above its train-locked threshold; otherwise the model returns current-only TTT.
8. Query/future frames are offline utility labels only. They never enter online TTT, memory addressing, transport, or routing.

Predicted geometry transport, DINO-only retrieval, learned history eviction, and a second TTT step are excluded from the primary path.

## Authoritative EXP-009 evidence

| Split/protocol | Routed utility | Harm | Comparator | Main inference |
|---|---:|---:|---|---|
| Train, source-safe leave-one-location-out | +1.933% | 9.17% | matched random +1.614% | Factorable observable utility address transfers to unseen locations. |
| Validation, unbounded one-shot | +2.642% | 5.83% | current-only TTT | Address/router transfer; capacity 8 fails. |
| Validation, selected reservoir-64 | +2.647% | 5.83% | random address +1.923% | Capacity 64 is the smallest registered passing value. |
| Locked test, reservoir-64 | **+2.088%** | **3.85%** | random address +1.602% | All eight terminal gates pass on 104 targets/22 components. |

Locked-test details:

- Current one-step TTT/base future loss ratio: **0.7018**.
- Reservoir-64 oracle top-K utility: **+3.356%**.
- Reservoir-64 routed utility: **+2.088%**; beneficial rate **65.38%**; acceptance **86.54%**.
- Same-bank random-address utility: **+1.602%**.
- Reservoir minus random component-bootstrap difference: **+0.475 points**, 95% CI **[+0.086, +0.924]**.
- Unbounded routed utility: **+2.068%**; reservoir retention: **100.98%**.
- FIFO-64 routed utility: **+2.068%**. Reservoir minus FIFO CI is **[-0.013, +0.059]** points, so retention-policy superiority is not supported.

The final compact result is `revisit3d/results/EXP-009/stage16_final_locked_test_v24.json`. The final artifact hash is `b5c7c1d0ce46ee078a1ea6d890a16e7bc5cc217a2a3777ee578d2b34ed760e98`.

## Hypothesis status

- H1 local reusable adaptation: **supported for proxy utility and aligned AbsRel, but not yet for consistent metric geometry**.
- H2-P predicted geometry carrier: **rejected**.
- H2-E predicted geometry primary router evidence: **rejected**.
- H3 online utility observability: **supported for the self-supervised future-loss target; metric-geometry observability is not supported**.
- H4-U learned utility addressing/routing: **supported for proxy ranking and aligned AbsRel only**; broad geometry utility and perfect negative-transfer rejection are not supported.
- H4-R separate learned risk classifier: **rejected in its current form**.
- H5 continual local-code consolidation: **supported only for bounded retention of proxy utility; open for consistent metric geometry**.
- H6 dynamic 4D extension: **open**.

## Experiment status

- EXP-001 — tttLRM fast-weight premise probes: completed, negative.
- EXP-002 — independent Revisit3D benchmark and objective health: completed.
- EXP-003 — compact/global/slot reuse: completed, negative for selectivity.
- EXP-004 — retrieval keys and vector routing: completed, partial/negative.
- EXP-005 — dense oracle 3D transport: completed as controlled evidence; old test split closed.
- EXP-006 — local visual plasticity atom and observable utility routing: completed.
- EXP-007 — bounded continual-bank feasibility: completed, partial.
- EXP-008 — true timestamp replay: completed on the original train split.
- EXP-009 — **completed** with a pre-locked 22-component terminal test; all registered gates passed.

## Evidence boundary and next research milestone

The demonstrated endpoint is a static nuScenes revisit benchmark using normalized future query-loss utility from frozen foundation geometry/tracking outputs. It is not yet an end-to-end claim for metric point-cloud accuracy, camera-pose correction, dynamic point tracking, arbitrary environments, or indefinite streams. One test component remains harmful, reservoir is not statistically superior to FIFO at capacity 64, and memory/query computation and wall-clock scaling still require a systems audit.

EXP-010 has now shown that the locked method significantly improves aligned AbsRel but worsens SILog and same-ray 3D endpoint error. The proxy-to-geometry bridge therefore failed its registered full gate. The immediate milestone is a train-only objective-health experiment; memory/router expansion is paused until one-step TTT improves all primary geometry metrics together.

The remaining priority order is:

1. replace the current absolute 3D track residual with a single symmetric frozen-track reprojection candidate and select the simplest healthy one-step objective on train only;
2. if healthy, refit the minimal atom/utility retrieval on train and run one locked validation before any new external test;
3. report absolute depth/point metrics and online latency/memory against no-TTT, current-only TTT, random address, and bounded/unbounded controls;
4. evaluate on a second dataset with a newly frozen component-disjoint protocol.

No EXP-009 test outcome may be used for further model selection.
