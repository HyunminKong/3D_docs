# Pre-framework validation round 2 — 2026-08-24

## Protocol

- Development: `nuscenes_revisit_dev.json` (20 train / 14 validation directional revisit episodes).
- Final confirmation: untouched original test split (6 directional episodes).
- Frozen VGGT supplies features and track/camera priors for controlled probes.  It is **not** a final reconstruction head.
- Online signal: frozen-track 3D consistency plus image-aware depth smoothness.
- Future query frames are never used by a TTT update or an online selection score; they are used only to measure utility / for the proposed future meta-training target.

## Claims that were rejected

1. **A global or low-dimensional update vector is a usable retrieval object.**
   It is not in this setup.  The learned 8-slot update rule collapsed: on validation, matched cosine was `0.999975` versus `0.999925` for negatives; after real 200-step training it collapsed further (`0.99999992` versus `0.99999983`).  Its centered update variance was 86.8–99.1% rank-1 and utility reranking was 0/14.
2. **Longer optimization fixes this collapse.**
   It does not.  A dataloader truncation initially made the 200-step run only 20 steps; after fixing it to cycle, the true 200-step result was more collapsed and had matched–foreign utility gap only `1.48e-7`.
3. **Raw local gradient similarity alone identifies the reusable memory.**
   Raw updates retained variation (positive–negative cosine gap `0.0567`), but their oracle utility top-1 was only 1/14; scaling the update by 1, 5, or 20 changed magnitude but not ranking.
4. **Episode-ID / one designated past traversal is the correct retrieval target.**
   It is not.  Several stored traversals can cover the same physical area.  The correct target is the past atom with the best *future utility*, not necessarily the originally paired A segment.

## Positive evidence

### Dense, correspondence-transported atom

The atom is a per-token log-depth residual.  It is updated with a normalized local TTT gradient and transported A -> A' through frozen-feature correspondence before current TTT.  On dev validation:

| Transport | matched - current | matched - foreign | utility top-1 |
|---|---:|---:|---:|
| Vector/raw state | about `-1.4e-5` | no usable gap | 1/14 |
| Dense visual transport (T=0.07) | `-2.29e-4` | `-3.30e-5` | 1/14 |
| Dense 3D-coordinate transport | `-3.36e-4` | `-9.69e-5` | 3/14 |
| Dense 3D + appearance transport | `-3.76e-4` | `-1.06e-4` | 4/14 |

The last version uses known camera poses only as an oracle upper bound.  It shows that spatially addressable geometry, rather than a global parameter direction, is the required reusable object.

### Current online geometry utility is informative

For every stored dense atom, we injected it into the current A' context, took one TTT step, and scored only current track consistency; the next query frames remained hidden.  Selecting the minimum online score over the full dev bank gave:

- oracle-best future utility: 5/14 (35.7%)
- oracle in online top-3: 8/14 (57.1%)
- mean future loss change versus current-only: `-6.05e-4`

On untouched test, the same fixed probe gave 3/6 exact oracle selections and a mean change of `-7.34e-4`; two of six selections still hurt.  Thus the signal is useful on average but **not safe enough to be a hand-designed gate**.

## Retrieval evidence

- Visual local-token key, top-5 + online rerank: mean future change `-4.51e-4` on dev.
- Oracle pose-map key, top-3 + online rerank: `-3.80e-4` on dev.
- Their naive union was worse (`-3.98e-4`), so simple score/key fusion is not justified.
- A scalar fusion of online loss and map distance overfit dev-train and failed on dev-val.  The selector must therefore consume richer candidate/current geometric features and be trained against a future utility target.

## Final design decision

The first actual framework should **not** be TTT-LRM plus a memory bank and should not store model-wide gradients.  It should be a new streaming geometry architecture with the following components:

1. **Frozen foundation backbone (initially VGGT; interface kept backbone-agnostic).**
   It produces dense tokens, initial point/depth/pose/track priors.  D4RT is a later backbone ablation, not the core contribution.
2. **Trainable geometry and plasticity head.**
   The head predicts depth/point residuals, tracking confidence, and pose correction.  Its fast state is a sparse set of 3D-addressed local plasticity atoms, not global weights or an 8-D/32-D vector.
3. **Geometry-only local TTT.**
   Track 3D consistency, multi-view point/depth consistency, pose consistency, and cycle consistency update only local atoms.  Each atom stores an anchor position, local feature key, low-rank residual/update code, uncertainty, and age/utility statistics.
4. **3D transport before reuse.**
   A retrieved atom is transported to the current token/anchor set with predicted world coordinates plus local appearance/correspondence.  Known poses in the present probe must be replaced by online pose/map estimates in the real method.
5. **Two-stage retrieval, then learned risk-aware utility.**
   A learned local relocalization key creates a small visual shortlist.  For each candidate, an online geometry probe is computed.  A trainable utility/risk head receives candidate atom, current geometry, correspondence, and probe statistics, and is meta-supervised by future frames during training.  It outputs a soft mixture / reject probability; a raw current-loss threshold is explicitly insufficient.
6. **Continual consolidation.**
   Write only high-confidence atoms whose predicted utility is positive; merge overlapping anchors in 3D and retain a compact low-rank code.  This is where continual learning belongs: preserving and consolidating *local adaptation experiences*, not protecting all backbone parameters with generic gradient projection.

## Required next implementation, now justified

Implement the trainable `3D plasticity atom head` and the future-utility/risk meta-objective before building a large memory bank.  The first ablation must compare global vector state, untransported local state, visual-only transport, geometry-only transport, and geometry+appearance transport under the same streaming protocol.
