# EXP-006 — Trainable 3D Plasticity Atoms and Risk-Aware Routing

## Status

In progress. Implementation protocol v2.2 was locked on 2026-08-25. Stage-0 train-only cross-fit passed; no official validation result exists yet.

## Question

Can a trainable local plasticity atom remain reusable after transport with predicted depth/pose, and can a learned current-context utility/risk router exploit it with less negative transfer than a current-loss heuristic?

## Hypotheses

- H2-P: predicted geometry plus appearance retains a measurable fraction of the oracle-coordinate transport benefit and outperforms visual-only transport.
- H4: learned utility/risk routing lowers clustered harm and future-utility regret without collapsing to reject-all.

## Authoritative protocol

[`EXP-006 Implementation Brief.md`](../EXP-006%20Implementation%20Brief.md), revision v2.2.

Key safeguards:

- Stage-0 predicted depth/pose/confidence health gate before atom training.
- K=5 utility-conditioned Stage-1 training; matched episode identity is not a label.
- First-order meta-gradient with detached discrete correspondence, inlier, Sim(3), and neighbor selection.
- Train-only neutral deadband, normalization, class weights, and grouped five-fold model selection.
- One-shot official validation; the exposed test split is rejected by the CLI.
- Undirected scene-pair/overlap-component grouped bootstrap inference.

## Data

- Manifest: `revisit3d/manifests/nuscenes_revisit_dev.json`
- Train: 20 directional episodes / 10 pairs / 8 overlap components.
- Validation: 14 directional episodes / 7 pairs / 2 overlap components.
- Test: 6 exposed directional episodes / 3 pairs / 1 component; closed and prohibited.

## Planned configuration and outputs

- Config: `configs/EXP-006_atom_utility.yaml`
- Geometry checkpoint: `revisit3d/checkpoints/exp006_geometry_bootstrap_v22.pt` (created; excluded from Git)
- Compact results: `revisit3d/results/EXP-006/`

## Planned primary metrics

- normalized future-loss change versus current-only TTT
- grouped-bootstrap confidence interval
- cluster and directional harm rates
- future-utility regret and accept/reject rate
- alignment validity, residual, inlier ratio, and transport retention
- runtime and peak GPU memory

## Result

### Stage-0 v2.1 identity-gate diagnostic — preserved failure

Result: `revisit3d/results/EXP-006/stage0_geometry_health_train_v21_identity_gate_failed.json`

- Passed 8/9 checks: finite/positive depth, view-0 identity, rotation, translation direction, translation scale, confidence correlation, and depth-residual gradient health.
- Cross-fit median rotation error: 0.446°.
- Cross-fit median translation-direction error: 4.598°.
- Cross-fit median scale-aligned translation error: 0.250.
- Component-mean confidence Spearman: 0.546.
- Predicted/identity track-loss ratio: 1.098, so the registered identity gate failed.
- A train-only counterfactual found the teacher/identity ratio was also greater than one. This identifies identity-pose degeneracy rather than failed pose distillation; see D011.

### Stage-0 v2.2 teacher-retention gate — passed

Result: `revisit3d/results/EXP-006/stage0_geometry_health_train_v22.json`

- All 9 corrected health checks passed on 20 directional train episodes grouped into 8 overlap components.
- Cross-fit median rotation error: 0.446°.
- Cross-fit median translation-direction error: 4.598°.
- Cross-fit median scale-aligned translation error: 0.250.
- Component-mean confidence Spearman: 0.546.
- Predicted/teacher track-loss ratio: 0.927, below the registered 1.05 ceiling.
- Teacher/identity ratio: 1.184, confirming the preserved identity degeneracy diagnostic.
- Positive depth fraction and healthy depth-residual gradient fraction: 1.0.

Stage 0 supports using the custom predicted pose/depth/confidence as the frozen EXP-006 base. It does not yet support H2-P or H4; those require Stage 1/2 and the unopened validation protocol.
