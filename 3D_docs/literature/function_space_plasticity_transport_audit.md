# Function-Space Plasticity Transport Audit

Last audited: 2026-08-26

## Motivation from completed experiments

EXP-040--060 repeatedly show that a local TTT code can reduce its online loss
but the same code is not reusable at another observation, even under oracle
pairing and correct 3D addressing. A code vector is a coordinate displacement;
its induced geometry depends on the observation-conditioned decoder Jacobian.
Transporting identical coordinates therefore does not transport an identical
change of the represented 3D function.

## Occupied foundations

| Direction | Representative work | Consequence |
|---|---|---|
| Functional continual learning | Functional Regularisation for Continual Learning (ICLR 2020) | Function-space preservation is established; it cannot be claimed generally as new. |
| Reparameterization-invariant optimization | Natural-gradient and higher-order invariance work | Parameterization invariance itself is established theory. |
| Gradient episodic memory/projection | GEM, OGD/GPM and later curvature-aware CL | Storing or constraining parameter gradients alone is occupied. |
| Generic continual TTA | CoTTA, EcoTTA, MECTA, CMF | Long-term TTA/forgetting is not the novelty. |
| 3D TTA | Point-TTA and GSDTTA | Test-time optimization for point-cloud recognition/registration is occupied and differs from streaming reconstruction. |
| Streaming TTT reconstruction | TTT3R and Mem3R | Adaptive recurrent updates and hybrid geometry/fast-weight memory are direct baselines, not the new object. |

## Candidate boundary

For a local adaptation coordinate `z` and observation-conditioned pointmap
`P_x(z)`, source code reuse assumes that `delta z` means the same thing at a
target. The candidate instead stores the first-order functional effect

`delta P_s = P_s(delta z_s) - P_s(0)`

at its predicted 3D support. At a revisit, predicted-3D correspondence
transports `delta P_s`; one gradient through the target decoder pulls that
desired displacement back to target coordinates. This is a discrete
push-forward/transport/pull-back operator for streaming 3D TTT experience.

The defensible claim is not a new natural gradient or generic CL method. It is
that **3D function-space transport resolves observation-dependent plasticity
coordinates in streaming pointmap adaptation**, demonstrated against the same
payload transported directly in code space and against spatial controls.

## Stop boundary

EXP-067 is a no-fit oracle-paired premise. Failure stops this operator without
training, Jacobian approximations, step-size tuning, routing, or validation.
Success authorizes one compact learned-address/deployment design, not a bank or
continual-consolidation stack.

## Primary sources

- Functional Regularisation for Continual Learning, ICLR 2020:
  <https://openreview.net/pdf?id=HkxCzeHFDB>
- Natural-gradient higher-order invariance, ICML 2018:
  <https://proceedings.mlr.press/v80/song18a.html>
- GEM, NeurIPS 2017: <https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning>
- EcoTTA, CVPR 2023:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Song_EcoTTA_Memory-Efficient_Continual_Test-Time_Adaptation_via_Self-Distilled_Regularization_CVPR_2023_paper.html>
- Point-TTA, ICCV 2023:
  <https://openaccess.thecvf.com/content/ICCV2023/html/Hatem_Point-TTA_Test-Time_Adaptation_for_Point_Cloud_Registration_Using_Multitask_Meta-Auxiliary_ICCV_2023_paper.html>
- GSDTTA, ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Wei_3D_Test-time_Adaptation_via_Graph_Spectral_Driven_Point_Shift_ICCV_2025_paper.html>
- TTT3R: <https://arxiv.org/abs/2509.26645>
- Mem3R: <https://lck666666.github.io/Mem3R/>
