# Gauge-Aware Reliability Literature Audit

Last checked: 2026-08-26

## Selection question

After EXP-060, what compact streaming-3D problem remains both consequential and
separable from current work without adding a memory bank, TTT optimizer, 4D
tracker, and multiple task heads to one paper?

## Rejected nearby directions

| Direction | Representative overlap | Decision |
|---|---|---|
| Reliability-controlled recurrent updates | ReCal3R, TTSA3R, SSR, MeMix | Rejected as the central claim. Learned reliability, selective update, and state mixing are directly occupied. |
| Long-stream cache or state compression | LONG3R, STAC, LongStream, OVGGT | Rejected. Memory scaling and spatio-temporal cache selection are active, crowded problems. |
| Generic pointwise uncertainty or calibration | Trust3R, *Uncertainty Quality of VGGT*, conformalized multi-view confidence calibration | Rejected as stated. Per-point evidential uncertainty, risk-coverage evaluation, and post-hoc calibration already exist. |
| Joint 4D reconstruction and tracking | SpatialTrackerV2, St4RTrack, C4D, Track4World | Rejected for the first paper. It broadens both architecture and evaluation and has strong direct competition. |

## Selected gap

Current pointmap reliability is usually represented per point. At the same
time, recurrent pointmaps possess scale/pose gauge ambiguity and long-stream
systems report scale and trajectory drift. Trust3R's local reliability tables
explicitly apply a separate Sim(3) alignment to each predicted pointmap before
MAE/RMSE and risk-coverage metrics. That is appropriate for local surface
quality, but it removes the correlated global error that a native streaming
system must still manage. MASt3R-SLAM likewise optimizes Sim(3) camera poses
because pointmap scale can be inconsistent. No audited work was found that
models these two uncertainties jointly as a shared gauge latent plus a local
surface residual for streaming pointmaps.

The provisional contribution boundary is therefore not "better confidence."
It is a hierarchical reliability model that exposes:

1. global/native-coordinate risk from a shared Sim(3) latent;
2. gauge-removed local surface risk from per-point residuals; and
3. the cross-point correlation induced by the shared latent.

## Closest sources

- Trust3R, *Uncertainty-Aware Evidential Learning for Reliable 3D Reconstruction*, 2026: <https://arxiv.org/abs/2605.19539>
- *Uncertainty Quality of VGGT: An Empirical Evaluation*, 2026: <https://arxiv.org/abs/2606.16479>
- *Conformalized confidence calibration for multi-view 3D reconstruction*, Pattern Recognition 2026: <https://www.sciencedirect.com/science/article/abs/pii/S0031320326012148>
- ReCal3R, *Reliability-Calibrated Recurrent 3D Reconstruction*, 2026: <https://arxiv.org/abs/2607.05356>
- TTSA3R, 2026: <https://arxiv.org/abs/2601.22615>
- SSR, 2026: <https://arxiv.org/abs/2603.14765>
- MeMix, 2026: <https://arxiv.org/abs/2603.15330>
- LONG3R, ICCV 2025: <https://openaccess.thecvf.com/content/ICCV2025/html/Chen_LONG3R_Long_Sequence_Streaming_3D_Reconstruction_ICCV_2025_paper.html>
- STAC, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/Wang_STAC_Plug-and-Play_Spatio-Temporal_Aware_Cache_Compression_for_Streaming_3D_Reconstruction_CVPR_2026_paper.html>
- LongStream, 2026: <https://arxiv.org/abs/2602.13172>
- OVGGT, 2026: <https://arxiv.org/abs/2603.05959>
- MASt3R-SLAM, CVPR 2025: <https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html>
- SpatialTrackerV2, ICCV 2025: <https://www.openaccess.thecvf.com/content/ICCV2025/html/Xiao_SpatialTrackerV2_Advancing_3D_Point_Tracking_with_Explicit_Camera_Motion_ICCV_2025_paper.html>
- St4RTrack, ICCV 2025: <https://www.openaccess.thecvf.com/content/ICCV2025/html/Feng_St4RTrack_Simultaneous_4D_Reconstruction_and_Tracking_in_the_World_ICCV_2025_paper.html>
- C4D, ICCV 2025: <https://www.openaccess.thecvf.com/content/ICCV2025/html/Wang_C4D_4D_Made_from_3D_through_Dual_Correspondences_ICCV_2025_paper.html>
- Track4World, 2026: <https://arxiv.org/abs/2603.02573>

## Remaining collision risk

The absence of an exact title/keyword match is not proof of novelty. Before a
head is trained, the audit must be repeated against probabilistic SLAM,
pose-uncertainty propagation, gauge-equivariant estimation, and correlated
dense prediction uncertainty. EXP-061 is deliberately only a phenomenon test.
