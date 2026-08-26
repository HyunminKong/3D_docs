# Cross-Clip 4D Consistency: 2023--2026 Collision Audit

Last updated: 2026-08-27

## Scope

This audit selects a paper premise after closing compact TTT adaptation-memory
reuse. It asks whether fixed-clip query-based 4D models produce a non-rigid
geometric discontinuity when the same physical query is decoded from two
overlapping video clips, and whether that discontinuity survives alignment
methods already occupied in the literature.

The local `Open-d4rt/` weakness study is only lead-generating evidence. It used
an unofficial reproduction and ten now-exposed ADT sequences. It cannot serve
as the registered paper premise. EXP-068 therefore freezes fresh sequences and
strong nuisance-removal controls before running either checkpoint.

## Closest work and occupied claims

### Query-based 4D reconstruction

- [D4RT (CVPR 2026 Best Paper)](https://d4rt-paper.github.io/) independently
  decodes a 3D point from `(u, v, t_src, t_tgt, t_cam)` and a single global
  clip representation. The paper uses learned discrete timestep embeddings and
  reports arbitrary spatio-temporal queries inside a fixed encoded video, but
  does not define cross-clip equivalence or a long-video stitching objective.
- [St4RTrack (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Feng_St4RTrack_Simultaneous_4D_Reconstruction_and_Tracking_in_the_World_ICCV_2025_paper.html)
  unifies reconstruction and world-frame tracking with paired dynamic
  pointmaps. It establishes the importance of world-coordinate 3D tracks, but
  not invariance of one query under overlapping clip contexts.
- [Point4Cast (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Point4Cast_Streaming_Dynamic_Scene_Reconstruction_and_Forecasting_CVPR_2026_paper.html)
  maintains a persistent spacetime representation that can be queried at past,
  present, and future times. A generic persistent 4D state or forecasting claim
  is therefore occupied.

### Sliding-window alignment and fusion

- [Geo4D (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Jiang_Geo4D_Leveraging_Video_Generators_for_Geometric_4D_Scene_Reconstruction_ICCV_2025_paper.html)
  fuses overlapping video-diffusion clips through group-wise pointmap, depth,
  camera, and scale alignment.
- [LASER (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Ding_LASER_Layer-wise_Scale_Alignment_for_Training-Free_Streaming_4D_Reconstruction_CVPR_2026_paper.html)
  converts offline geometry models to streaming inference by aligning adjacent
  windows and explicitly shows that one global Sim(3) is insufficient when
  scene layers have different scales.
- [V-DPM (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Sucar_V-DPM_4D_Video_Reconstruction_with_Dynamic_Point_Maps_CVPR_2026_paper.html)
  uses overlapping windows and bundle-adjustment-style fusion for long videos.

Consequently, the project cannot claim sliding windows, overlap alignment,
Sim(3) stitching, layer-wise scale correction, global bundle adjustment, or
generic long-video 4D reconstruction as novel.

### Relational and rigidity regularization

- [4DRegSDF (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Choe_Spacetime_Surface_Regularization_for_Neural_Dynamic_Scene_Reconstruction_ICCV_2023_paper.html)
  imposes local rigidity on deformable surfaces.
- [Shape of Motion (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html)
  includes a distance-preserving loss between dynamic Gaussians and their
  neighbors.
- Scene-flow work has long used local smoothness, object rigidity, cycle
  consistency, and pairwise distance constraints. A generic rigidity or
  pairwise loss is not a contribution by itself.

### Evaluation

- [TAPVid-3D](https://arxiv.org/abs/2407.05921) and the world-frame protocol
  used by St4RTrack/D4RT primarily score aligned pointwise location and
  visibility accuracy. They do not directly require two overlapping encoder
  contexts to assign the same geometry to an identical physical query.
- A new metric is insufficient alone. It must expose a large reproducible
  residual that is not removed by the strongest occupied alignment family and
  must predict a correction opportunity beyond ordinary pointwise error.

## Candidate comparison

| Candidate | Empirical lead | Collision risk | Implementation burden | Decision |
|---|---:|---:|---:|---|
| Cross-clip query-equivalence residual beyond alignment | Strong: prior exposed probe showed 5--7x structural degradation | Medium-high: LASER/Geo4D/V-DPM | Low for no-fit; one loss if premise passes | **Selected for EXP-068** |
| Pairwise motion-compression loss | Strong on exposed PStudio | High: rigidity/scene-flow losses are mature | Medium, requires fine-tuning | Backup only |
| Continuous physical-time query | Architecturally clean | Medium; D4RT already trains with random temporal stride and visual evidence may resolve cadence | Medium-high | Not selected |
| Delayed state repair after corrupted observations | Plausible | High: RayMap3R, TTSA3R, SSR, outlier-view rejection | Medium-high | Rejected |
| Generic long-video memory/window stitching | Strong practical need | Directly occupied | High | Rejected |

## Selected novelty boundary

The only provisionally defensible question is narrower than window alignment:

> For the exact same physical point query in the overlap of two clips, does an
> independent-query 4D foundation model return geometries that remain
> relationally inconsistent after held-out global Sim(3) and an oracle
> depth-layer alignment? Is the residual materially larger than an adjacent
> one-frame context shift while ordinary pointwise APD changes little?

If the answer is no, this branch closes because existing alignment work covers
the failure. If yes, the possible paper contribution is the conjunction of:

1. a query-equivalence formulation specific to independently decoded 4D
   foundation models;
2. a source-safe cross-clip relational diagnostic beyond gauge/layer scale;
3. one offline equivalence-consistency loss with no query-query attention,
   memory bank, inference optimization, or added prediction head;
4. preservation of pointwise tracking and reconstruction accuracy.

EXP-068 tests only items 1--2. No method is authorized before its complete gate
passes on fresh premise sequences.
