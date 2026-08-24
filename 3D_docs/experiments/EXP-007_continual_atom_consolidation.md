# EXP-007 — Continual Atom Bank and Consolidation

Status: **Completed on the train-only causal pseudo-stream; H5 partially supported.**

## Question

Can a causally written, capacity-bounded bank of visually addressable plasticity atoms retain the future-utility benefit of an unbounded bank without increasing negative transfer?

## Hypothesis

H5: adaptation-regime-aware consolidation can preserve most routed revisit utility with sublinear memory growth. Place identity alone is not the consolidation target.

## Leakage boundary

- Stage 0 and all architecture selection use the expanded train split only.
- An atom is retrievable only after its context has appeared in the simulated stream.
- Query/future frames measure utility only and never enter writes, keys, router features, merge, eviction, or selection.
- EXP-006 validation is closed to EXP-007 tuning; the previously exposed test split remains prohibited.
- Stream order, capacities, baselines, and success criteria must be fixed before any future EXP-007 holdout is opened.

## Protocol

- Data: 76 expanded-train directional episodes, 19 physical-overlap components.
- Cross-fitting: five component-safe folds; an atom head/router is trained outside the held-out fold and each bank remains fold-local.
- Stream: ten deterministic pseudo-orders per fold. A/B contexts are written before an event, A′ is written only after its query utility is measured.
- Leakage: future/query frames are offline utility labels or delayed history only. They never enter same-event TTT, keys, writes, retrieval, or routing.
- Closed data: EXP-006 validation and the exposed six-episode test split were never accessed.
- Primary bank: capacity 8, appearance top-5 retrieval, one selected memory, predicted-utility history eviction.
- Primary deployable bucket key: frozen VGGT token sets projected by PCA fit outside the held-out fold, followed by an OOF logistic pair classifier.

The orders are synthetic rearrangements of selected revisit episodes. They test causal memory mechanics but are not a real chronological streaming claim.

## Required controls

- no memory / one current TTT step;
- unbounded all-write bank;
- FIFO;
- uniform reservoir;
- appearance-only diversity;
- place/scene deduplication;
- oracle capacity subset upper bound;
- random retrieval and visual mean where computationally comparable.

## Metrics

- normalized future utility and regret;
- directional and component deadband harm;
- raw-sign harm;
- accept/reject rate;
- top-K recall of the all-bank utility oracle;
- retained utility relative to unbounded router and oracle;
- records, bytes, candidate comparisons, wall time, and peak GPU memory;
- write, merge, eviction, and reactivation counts.

## Results

### 1. Capacity is not the main bottleneck; distractor scaling is

In leakage-safe OOF simulation, unbounded all-write top-5 routing achieved +0.02091 utility with 7.37% harm. Capacity-8 controls were better: FIFO +0.02403/7.37%, reservoir +0.02528/7.50%, scene-latest +0.02615/6.32%, and appearance-diversity +0.02515/8.16%. Scene-latest minus unbounded was +0.00466 with component-bootstrap CI [0.00189, 0.00725], while the bank fell from maximum 83 records to 8.

This is not evidence that FIFO is continual learning. The fixed EXP-006 router had been trained on K=5 pools; larger banks changed its winner distribution. Raw winning score correlated negatively with realized utility in the bank-aware analysis (Spearman −0.457).

### 2. Utility history has signal, but does not by itself solve safety

At capacity 8, predicted history reached +0.02782 utility/6.71% harm and hybrid history +0.02808/6.71%, versus scene-latest +0.02615/6.32%. They improved utility but failed the registered no-extra-harm requirement. The future-coverage oracle reached +0.02940, leaving +0.00325 utility headroom.

Bank-aware acceptance calibration and set-normalized reranking did not repair this boundary. The calibrated primary reached +0.02702/6.71%; the listwise primary reached +0.02789/6.71%. Both beat scene-latest utility but retained more harm. The current-objective gate was safe (0.13% harm) only by discarding most benefit (+0.01025 utility).

### 3. Oracle buckets show that utility-aware consolidation is possible

With one record per oracle scene, delayed top-5 utility history reached +0.02674 utility/6.18% harm. Its paired component-bootstrap difference over scene-latest was small but positive: +0.00043, CI [+0.00006, +0.00086]. Scene ID is an oracle grouping control, not deployable input.

### 4. The local transport key is not a consolidation key

- Pooled learned-key bucket: +0.02260 utility, 7.76% harm, 84.5% oracle-scene retention.
- Learned atom token-set bucket: OOF same-scene AUC 0.654, +0.02280 utility, 8.55% harm, 85.3% retention.

Both failed. Meta-training a token key for local code transport does not preserve traversal-level place separability.

### 5. A separate frozen token key is a promising consolidation key

The strict fold-local-PCA control reached OOF same-scene AUC 0.650. Despite weak semantic precision, the capacity-8 predicted-history bank achieved:

- mean utility **+0.02618**;
- deadband harm **8.03%** versus appearance-diversity **8.16%**;
- **97.9%** of oracle-scene utility;
- maximum **8** records and 79.7% causal-oracle top-5 recall.

Its component-bootstrap difference over appearance diversity was uncertain (+0.00035, CI [−0.00166, +0.00225]). However, a registered matched null that shuffled pair scores within each fold while preserving their distribution placed the observed utility at the **99.4th percentile** of 1,000 permutations (one-sided p=0.00699). Thus useful key association exists beyond arbitrary bank shrinking. The observed harm was not lower than the mean permutation harm, so the key is supported for consolidation/prefiltering, not for safety gating.

## Conclusion

H5 is **partially supported**. A capacity-bounded atom bank can retain reuse utility, causal adaptation history is informative, and a key separate from the learned transport key beats a merge-rate-matched null. The result does not establish a final continual system because:

- all architecture selection is on a pseudo-stream built from train episodes;
- same-scene AUC and precision remain weak;
- the fixed K=5 router is miscalibrated under larger candidate sets;
- learned rejection and lower negative transfer remain unsolved;
- capacity 8 is benchmark-specific, not a universal constant.

## Architecture decision

Use two distinct address spaces:

1. **local transport address** — learned per-token appearance key for moving an 8-D plasticity code into the current token frame;
2. **long-term consolidation address** — frozen, separately normalized foundation token-set key for bank bucketing and candidate prefiltering.

The provisional bank uses predicted-utility history for retention and bounded residual reuse for damage control. Consolidation similarity must never be treated as the final accept/reject safety decision.

## Artifacts

- Configs: `configs/EXP-007_*.yaml`
- Scripts: `revisit3d/scripts/*exp007*.py`
- Compact result: `revisit3d/results/EXP-007/summary_v21.json`
- Full local tables are intentionally Git-ignored and their hashes are recorded in the compact summary.

## Next experiment

EXP-008 must replace pseudo-orders with real capture timestamps and unique context writes. It should first reuse the frozen utility table on train only, then define a new scene/location-disjoint paper-scale benchmark before any new validation is opened.
