# Query-Integrity Evaluation for 4D Reconstruction: Collision Audit

Last updated: 2026-08-27

## Question

Can a pointwise 3D tracking metric prefer a prediction produced under a larger
irrelevant query-context change even when that change causes a larger
counterfactual geometry inconsistency? The candidate contribution is not a new
rigidity metric in isolation. It is an evaluation of whether semantically
equivalent queries form a consistent equivalence class.

## Empirical lead and source-safety boundary

EXP-068 was registered around an absolute APD-blindness threshold and failed
that complete gate. Its exposed premise rows nevertheless reveal a different
phenomenon that was not used in the v1.0 gate: the 16-frame context shift has a
larger aligned structural residual than the one-frame shift in 16/16 sequences,
yet its APD is better than the reference context in 9/16 sequences. Signed APD
gain and the structural residual have Pearson correlation `0.0096`.

This is lead-generating evidence only. H25 and EXP-069 freeze a ranking test
before opening the 11 NPZ files assigned to EXP-068's unused validation role.
Those files cease to be available for H24 after this reassignment. The 12-file
terminal role remains unopened.

## Closest primary work

- [D4RT](https://arxiv.org/html/2512.08924) defines the independent query
  `(u,v,t_src,t_tgt,t_cam)` and uses differently referenced point sets to
  recover camera extrinsics through Umeyama alignment. Its tracking evaluation
  reports pointwise APD/EPE, not invariance under an equivalent query context.
- [UniQuery4R](https://arxiv.org/html/2608.17283), released on 18 August 2026,
  removes D4RT's fixed temporal embeddings and predicts camera parameters per
  view. It still evaluates dynamic points with WorldTrack APD/EPE after global
  scale alignment. It therefore narrows the architectural opportunity but does
  not test counterfactual query integrity.
- [TAPVid-3D](https://arxiv.org/abs/2407.05921) and WorldTrack evaluate each
  tracked point against ground truth with thresholded pointwise accuracy. They
  do not compare two semantically equivalent ways of asking for the same 4D
  point.
- [PDI-Bench](https://pdi-bench.github.io/) evaluates structural rigidity in
  generated videos. This occupies a generic claim that pairwise-distance
  stability is itself new, but it does not define query equivalence or test
  whether a point-tracking metric rewards a less integral query context.
- Geo4D, LASER, and V-DPM align/fuse temporal windows. They occupy window
  stitching and layer-scale correction; they do not establish that pointwise
  model ranking agrees with counterfactual query integrity.

## Occupied claims

The project may not claim novelty for APD/EPE, pair-distance rigidity,
overlapping windows, Sim(3) alignment, layer-wise alignment, generic cycle
consistency, or a generic relational loss. The August 2026 UniQuery4R result
also makes replacing D4RT's temporal embeddings or adding a separate camera
head an occupied/weak immediate direction.

## Provisional boundary

The only defensible premise is a ranking contradiction:

> A larger irrelevant clip-context change produces a universally larger
> held-out non-gauge residual, while ordinary APD does not penalize it and often
> rewards it. Therefore pointwise accuracy and query integrity are distinct
> axes of 4D reconstruction quality.

This is stronger than saying that APD changes by less than an arbitrary amount.
It requires the pointwise metric to select the structurally less stable context
often enough to matter, a non-negative mean signed APD gain, and near-zero
association between APD gain and structural damage on untouched sequences.

If EXP-069 fails, the evaluation branch closes. If it passes, it authorizes a
broader multi-model/dataset feasibility audit. It does not yet authorize a loss,
benchmark paper, or architecture.
