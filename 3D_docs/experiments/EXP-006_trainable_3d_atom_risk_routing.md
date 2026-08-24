# EXP-006 — Trainable 3D Plasticity Atoms and Risk-Aware Routing

## Status

In progress. Implementation protocol v2.3 is active as of 2026-08-25. Stage-0 train-only cross-fit passed; no official validation result exists yet.

## Question

Can a trainable local plasticity atom remain reusable after transport with predicted depth/pose, and can a learned current-context utility/risk router exploit it with less negative transfer than a current-loss heuristic?

## Hypotheses

- H2-P: predicted geometry plus appearance retains a measurable fraction of the oracle-coordinate transport benefit and outperforms visual-only transport.
- H4: learned utility/risk routing lowers clustered harm and future-utility regret without collapsing to reject-all.

## Authoritative protocol

[`EXP-006 Implementation Brief.md`](../EXP-006%20Implementation%20Brief.md), revision v2.3.

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

### Stage-1 v2.2 cross-view-scale diagnostic — preserved failure

Result: `revisit3d/results/EXP-006/stage1_predicted_transport_health_train_v22_crossview_scale_failed.json`

- Predicted Sim(3) was evaluated only on the 20 directional train episodes; validation and the closed test split were not accessed.
- Matched A→A' alignment validity was 0.70, below the 0.80 preflight gate.
- Current-context→future-query alignment validity was 1.00 and every valid transport was finite.
- The failure was traced to pooling all eight overlapping views before median 8-NN scale estimation. Cross-view duplicate observations made the metric-normalization bandwidth artificially small.
- A train-only, single-variable counterfactual using within-view 8-NN yielded matched validity 1.00 without changing any Sim(3) threshold. D012 and protocol v2.3 register this definition before validation.

### Stage-1 v2.3 predicted-transport health gate — passed

Result: `revisit3d/results/EXP-006/stage1_predicted_transport_health_train_v23.json`

- All 20/20 matched A→A' alignments and 20/20 current-context→future-query alignments were valid.
- Median matched correspondence count was 188.5, median inlier ratio 0.925, and median normalized residual 0.889 local spacings.
- All valid geometry+appearance transports were finite.
- Distant-B validity was 0.90 and deterministic foreign-candidate validity was 0.467. Alignment validity is therefore a geometric feasibility mask, not a relevance or utility label; candidate selection must remain utility-conditioned as specified by D005/D006.
- The diagnostic used train only. Validation and the closed test split remain unopened.

This gate supports proceeding to atom meta-training with predicted geometry. It does not yet establish H2-P future utility or H4 routing safety.
