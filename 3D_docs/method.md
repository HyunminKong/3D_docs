# Current Method Specification

## Status

The compact EXP-015 + EXP-019 architecture below is the most recent frozen
candidate, but EXP-020 rejected it as the final paper model. It is retained as
the reference implementation and ablation anchor; it must not be presented as
a validated broad reconstruction method.

## Frozen reference candidate

```text
streaming RGB context
        ↓
frozen VGGT/FastVGGT dense features and geometry evidence
        ↓
8-D per-token plasticity code initialized at zero
        ↓
one local-code step on L_track3D, eta=0.0125
        │
        ├── current-only fallback
        │
64-D pooled descriptor → one factorized Ridge utility-MIPS top-5
        ↓
visual token correspondence transports five stored codes
        ↓
positive current-geometry agreement reranking; coarse top-1 fallback
        ↓
z_out = clamp(z_current + 0.10 z_memory, -1, 1)
        ↓
depth / point readout

write after prediction → deterministic reservoir, capacity 64/location
```

The only learned inference-time additions to the frozen foundation are the
plasticity head and one linear utility address. Agreement reranking, fallback,
and reservoir retention are parameter-free. There is no fine router, risk head,
learned decision threshold, learned eviction, pose update, second TTT step, or
dynamic state.

## Online objective

The deployed TTT objective is exactly one absolute frozen-track 3D consistency
loss:

\[
\mathcal L_{\mathrm{TTT}}=\mathcal L_{\mathrm{track3D}}.
\]

Only the local code is differentiated, for one step. Query/future observations
are never online inputs. EXP-011 established that this objective and step size
can be metric-healthy; EXP-020 showed that metric health was not preserved by
the later atom meta-refit.

## Offline atom objective used by the rejected candidate

EXP-015 used three unweighted readouts of the same track3D signal:

1. normalized current-only future quality;
2. absolute best-reuse future quality among five train-only candidates;
3. a softplus ranking of best reuse against stop-gradient current quality.

No key contrastive, harmful-code neutralization, centering, smoothness, or code
norm loss was used. Despite strong OOF proxy headroom, EXP-020 indicates that
this proxy-oriented objective did not preserve broad LiDAR metric alignment.

## Plasticity record and causal contract

Each record contains a local visual key, an 8-D per-token code, a pooled 64-D
descriptor, current-observable adaptation statistics, timestamp, and stream
partition. A target is predicted before its record is written. Future/query
frames may produce offline meta labels and evaluation metrics only. Source-safe
folds remove held physical entities from both target and memory-source training
rows.

## Utility address

For current/source descriptors `c_t,c_i`, a Ridge score over

```text
[c_t, c_i, c_t-c_i, c_t*c_i]
```

factorizes exactly into maximum inner-product search. Five candidates are
retrieved. A candidate with positive coarse score and positive current-geometry
agreement can reroute the coarse winner; otherwise the coarse top-1 is used.
Both decisions use semantic zero rather than calibrated thresholds.

## Theoretical rationale

For an `L`-smooth future loss `F` and transported residual `delta`,

\[
F(z)-F(z+\delta)\ge
-\langle\nabla F(z),\delta\rangle-\frac{L}{2}\lVert\delta\rVert^2.
\]

Transport and utility addressing aim to make the first-order term favorable;
the fixed residual and clamp bound second-order damage. This motivates the
method but is not a worst-case safety theorem. EXP-020's 33% proxy harm confirms
that the bound alone is insufficient without a utility target calibrated to the
paper endpoint.

## Allowed next-method change

No change is authorized yet. After a new independent benchmark is identified,
the next paper-scope decision may authorize at most one metric-aligned offline
meta signal. The online objective must remain one loss and the inference graph
must not gain another learned module unless a new decision explicitly replaces,
rather than stacks onto, an existing component.

## Claim boundary

Supported: local transported corrections contain reusable proxy information;
utility addressing can beat appearance and matched random controls; bounded
causal storage can retain this proxy effect; aligned AbsRel improvement over
current TTT exists in EXP-020.

Unsupported: broad metric reconstruction improvement, reliable negative
transfer rejection, pose/4D claims, reservoir superiority, universal capacity,
or generalization to a second dataset/backbone.
