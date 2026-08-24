# Current Method Specification

## Final static-revisit architecture: utility-addressed local plasticity memory

```text
streaming RGB context x_t
        ↓
frozen VGGT/FastVGGT encoder and geometry evidence
        ↓
custom spatial plasticity head
        ↓
one current-only TTT step on 8-D per-token code z_t
        │
        ├────────────────────────────── current-only fallback
        │
pooled transport descriptor c_t
        ↓
exact 64-D utility-MIPS address ── top K=5 from causal bank
        ↓
visual token correspondence transports each stored local code
        ↓
observable utility router selects one candidate or rejects all
        ↓
z_out = clamp(z_t + 0.10 z_memory, -1, 1)
        ↓
depth / point readout

causal write after prediction
        ↓
deterministic reservoir bank, capacity 64 per location stream
```

The foundation, plasticity-head weights, utility address, and router are frozen online. Only the local fast code is updated by TTT.

## Plasticity memory object

Each record contains:

- a normalized per-token visual key;
- an 8-D per-token fast code obtained by one current-only TTT step;
- a pooled 64-D transport descriptor;
- predicted 3D anchor, scale, and confidence as metadata;
- current-only objective and track-consistency statistics;
- timestamp and stream partition needed for causal replay and retention.

Predicted `xyz` is not used to transport the code or as a primary address. The local key is used for token-level visual transport; the pooled descriptor is used for long-term candidate addressing. This separation is necessary because a useful correspondence key and a useful adaptation-utility address need different invariances.

## Online TTT objective

The online objective uses only the current context:

- frozen-track 3D consistency;
- edge-aware depth smoothness;
- bounded code regularization.

Exactly one step with the locked step size is applied. A second step was harmful in EXP-006. Query/future frames are read-only: they produce meta-training utility labels and evaluation metrics but never enter adaptation, memory retrieval, transport, or router features.

## Utility address

Let `c_t` and `c_i` be pooled current and source transport descriptors. The pair scorer is a Ridge model over

```text
[c_t, c_i, c_t - c_i, c_t * c_i].
```

Its linear score is algebraically compiled into a 64-D maximum-inner-product search score, so retrieval does not enumerate future utility or run code transport across the full bank. The compilation error at lock time was below `3.5e-9`. The scorer was trained on train-only future utility and validated with source-entity-safe leave-one-location-out folds.

Frozen DINOv2 place descriptors are retained only as a negative/control representation. They achieved strong place-overlap AUC but were worse than matched random retrieval for causal plasticity utility.

## Transport and utility routing

For each of the K=5 addressed memories:

1. learned visual correspondence reads the source code at current tokens;
2. the transported code is provisionally combined with current TTT at fixed strength 0.10;
3. observable current/source objective histories, descriptor interactions, code statistics, and visual-transport statistics form the router input;
4. a frozen `StandardScaler → PCA(16) → Ridge(alpha=1)` model predicts utility;
5. the best candidate is accepted only above the train-locked threshold; otherwise the system returns current-only TTT.

No query/future feature is present. Predicted Sim(3) alignment is an ablation and does not gate visual transport.

## Continual bank

- Stream partition: official location for the current benchmark.
- Causal order: true nuScenes capture timestamp; each target is evaluated before its own write.
- Retention: deterministic reservoir sampling.
- Capacity: 64 records per stream partition.
- Retrieval: utility-MIPS K=5.
- Write: after current prediction/adaptation.

Capacity 64 is the smallest value passing the registered validation gate among `{8,16,32,64}`. It is a benchmark-selected operating point, not a universal constant. Learned utility-history eviction was rejected. Reservoir is retained as a simple unbiased policy, but final-test reservoir and FIFO utilities were statistically indistinguishable; the supported claim is bounded retention, not reservoir superiority.

## Evidence boundary

The terminal EXP-009 test contains 104 unique target contexts from 117 canonical directions, grouped into 22 physical overlap components across four locations. Reservoir-64 produced +2.088% normalized future utility versus +1.602% for same-bank random addressing; the paired component 95% CI was [+0.086, +0.924] percentage points. It retained 100.98% of unbounded addressed-bank utility and passed all pre-registered gates.

The current output metric is a normalized future depth/point query-loss proxy derived from frozen reconstruction and tracking evidence. The method has not yet established end-to-end metric depth, point-cloud, pose, dynamic tracking, cross-dataset, indefinite-stream, or wall-clock claims. Those are EXP-010 and later milestones. EXP-009 test is terminal and cannot be used for further method selection.
