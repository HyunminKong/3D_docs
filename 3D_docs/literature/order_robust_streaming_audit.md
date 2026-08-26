# Order-Robust Streaming 3D Literature Audit

Last checked: 2026-08-26

## Candidate distinction

The candidate does not claim that sets should be permutation invariant in
general. That is classical and is used by set-based SfM and multi-view
Transformers. It asks whether a *causal recurrent reconstruction operator* can
be trained so that assimilating the same static evidence is approximately path
independent, while retaining a single chronological pass at deployment.

## Collision map

| Area | Representative work | Boundary |
|---|---|---|
| Offline order-free reconstruction | Deep Permutation Equivariant SfM; VGGT-family and PanoVGGT set models | Full-set joint inference is not the claimed contribution. |
| Adaptive recurrent updates | TTSA3R, PAS3R, ReCal3R, FILT3R | These change update magnitude/gain from reliability, motion, or variance; do not explicitly minimize swapped-path commutators. |
| Temporal state regularization | SSR | Regularizes a state trajectory on a Grassmannian/self-expressive model; not fixed-evidence path independence. |
| Long-stream memory | LONG3R, LongStream, STAC | Memory selection/compression/scaling are excluded from the proposed method. |
| Order ensembling/global alignment | CUT3R `demo_ga.py` | The released demo constructs forward/backward permutations before global alignment. This acknowledges practical dependence but pays repeated inference and does not learn a causal order-robust update. |
| Recurrent permutation regularization | SIRE, NeurIPS 2020; Learnable Commutative Monoids, LoG 2022 | Direct generic collision: pairwise swapped paths are already regularized toward equal latent states. A generic commutator loss is not novel. |

## Provisional novelty boundary

1. measure a *geometry-decoded* recurrent commutator at a fixed query;
2. show it explains absolute 3D risk beyond normalized latent-state distance;
3. formulate path consistency in the geometry quotient that removes only the
   permitted monocular scale, not arbitrary latent differences;
4. keep chronological causal inference unchanged.

The exact phrase search found no direct streaming pointmap paper with this
conjunction. That is not proof of novelty. A passing premise must be followed by
a broader audit of order-robust RNNs, path-independent neural operators,
commutative aggregation, and incremental SfM before training. The generic
commutator claim is already rejected by SIRE.

## Primary sources

- CUT3R, *Continuous 3D Perception Model with Persistent State*, CVPR 2025:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html>
- Deep Permutation Equivariant SfM, ICCV 2021:
  <https://openaccess.thecvf.com/content/ICCV2021/html/Moran_Deep_Permutation_Equivariant_Structure_From_Motion_ICCV_2021_paper.html>
- PanoVGGT, CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Guo_PanoVGGT_Feed-Forward_3D_Reconstruction_from_Panoramic_Imagery_CVPR_2026_paper.html>
- TTSA3R: <https://arxiv.org/abs/2601.22615>
- PAS3R: <https://arxiv.org/abs/2603.21436>
- ReCal3R: <https://arxiv.org/abs/2607.05356>
- FILT3R: <https://arxiv.org/abs/2603.18493>
- SSR: <https://arxiv.org/abs/2603.14765>
- LONG3R, ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html>
- LongStream, CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Cheng_LongStream_Long-Sequence_Streaming_Autoregressive_Visual_Geometry_CVPR_2026_paper.html>
- STAC, CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Wang_STAC_Plug-and-Play_Spatio-Temporal_Aware_Cache_Compression_for_Streaming_3D_Reconstruction_CVPR_2026_paper.html>
- SIRE, *Regularizing Towards Permutation Invariance in Recurrent Models*,
  NeurIPS 2020:
  <https://papers.nips.cc/paper_files/paper/2020/hash/d58f36f7679f85784d8b010ff248f898-Abstract.html>
- *Learnable Commutative Monoids for Graph Neural Networks*, LoG 2022:
  <https://proceedings.mlr.press/v198/ong22a.html>
