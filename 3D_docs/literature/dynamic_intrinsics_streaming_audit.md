# Dynamic-Intrinsics Streaming 3D Audit (2023--2026)

Last audited: 2026-08-26

## Candidate problem

Real video can change focal length and field of view during capture through
optical zoom, focus breathing, stabilization, crop, or multi-camera switching.
The candidate question is not merely whether intrinsics can be estimated. It is
whether a transient camera-geometry change is written into a persistent neural
3D state and continues to damage later clean-view geometry after the camera
returns to its prior regime.

## Occupied directions

| Direction | Representative work | Consequence for this project |
|---|---|---|
| Video self-calibration | DroidCalib, ICCV 2023 | Differentiable self-calibrating bundle adjustment already estimates intrinsics from video. “Online focal estimation” alone is not novel. |
| Single-image calibration | GeoCalib, ECCV 2024; AnyCalib, ICCV 2025 | Learned geometric calibration and camera-model-agnostic ray/FoV prediction are occupied. A new focal head alone is not a contribution. |
| Dynamic-intrinsics data | InFlux, NeurIPS 2025 Datasets & Benchmarks; InFlux++ 2026 | Per-frame zoom/focus calibration is an established benchmark problem. InFlux++ also releases synthetic pose/depth for a subset, making downstream geometry evaluation feasible. |
| Prior-conditioned reconstruction | Pow3R, CVPR 2025; G-CUT3R, 2025 | Feeding known intrinsics/depth/pose through a new encoder is occupied. A generic camera-prior branch is not novel. |
| Test-time prior use | TCO-VGGT, CVPR 2026 | Treating camera priors as test-time constraints for multiview Transformers is occupied and computationally heavy. “Use intrinsics in TTT” is not sufficient positioning. |
| Camera-agnostic reconstruction | CAM3R, 2026 | Decomposing rays and radial distance for mixed pinhole/fisheye/panoramic imagery is occupied in feed-forward reconstruction. |
| Generic recurrent update robustness | ReCal3R, FILT3R, PAS3R, MeMix, 2026 | Generic reliability gates, Kalman filtering, pose-aware rates, and sparse writes are crowded. A generic noisy-frame gate is not allowed. |

## Remaining defensible boundary

The audited works do not isolate the following causal failure in a recurrent
pointmap foundation model:

1. keep the scene, clean history, later clean query, and carrier fixed;
2. change only one intermediate view to a physically valid focal/FoV
   reparameterization;
3. distinguish immediate image degradation from persistent state-write damage
   using `update=false`, resampling, and missing-periphery controls;
4. measure downstream clean-query 3D geometry after an additional clean state
   update.

If the premise exists, the paper claim must remain **calibration-shock-resilient
state writing**, not camera calibration, generic prior fusion, generic TTT, or
camera-agnostic reconstruction. A later method must be compact, causal, and
directly remove the persistent excess over matched controls. It may use known
or independently predicted per-frame camera geometry, but must report those two
settings separately.

## Collision risks

- A method that simply encodes `K` and fuses it with RGB collides with Pow3R and
  G-CUT3R.
- A decoder/network optimization with camera-prior penalties collides with
  TCO-VGGT.
- A learned ray head collides with AnyCalib and CAM3R unless its role is
  specifically temporal state canonicalization and the reconstruction benefit
  is demonstrated.
- A scalar write gate collides with ReCal3R/PAS3R/FILT3R unless it separates
  camera-coordinate change from observation reliability.
- Center-crop/resize alone is not sufficient evidence because it changes image
  bandwidth and visible support. EXP-065 therefore freezes both controls.

## Primary sources

- DroidCalib, ICCV 2023:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Hagemann_Deep_Geometry-Aware_Camera_Self-Calibration_from_Video_ICCV_2023_paper.html>
- GeoCalib, ECCV 2024:
  <https://eccv.ecva.net/virtual/2024/poster/1349>
- InFlux, NeurIPS 2025:
  <https://proceedings.neurips.cc/paper_files/paper/2025/file/8a8eca190088852067b4e8cc1b907122-Paper-Datasets_and_Benchmarks_Track.pdf>
- AnyCalib, ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Tirado-Garin_AnyCalib_On-Manifold_Learning_for_Model-Agnostic_Single-View_Camera_Calibration_ICCV_2025_paper.html>
- Pow3R, CVPR 2025:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Pow3R_Empowering_Unconstrained_3D_Reconstruction_with_Camera_and_Scene_Priors_CVPR_2025_paper.html>
- G-CUT3R:
  <https://arxiv.org/abs/2508.11379>
- TCO-VGGT, CVPR 2026:
  <https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Learning_3D_Reconstruction_with_Priors_in_Test_Time_CVPR_2026_paper.html>
- CAM3R:
  <https://arxiv.org/abs/2603.22631>
- InFlux/InFlux++ methodology and release:
  <https://influx.cs.princeton.edu/methodology/>
  <https://influx.cs.princeton.edu/data/>
- ReCal3R:
  <https://arxiv.org/abs/2607.05356>
- FILT3R:
  <https://jinotter3.github.io/FILT3R/>

