# EXP-006 — Trainable 3D Plasticity Atoms and Risk-Aware Routing

## Status

Planned. Implementation protocol v2 was locked on 2026-08-25; no EXP-006 training or validation result exists yet.

## Question

Can a trainable local plasticity atom remain reusable after transport with predicted depth/pose, and can a learned current-context utility/risk router exploit it with less negative transfer than a current-loss heuristic?

## Hypotheses

- H2-P: predicted geometry plus appearance retains a measurable fraction of the oracle-coordinate transport benefit and outperforms visual-only transport.
- H4: learned utility/risk routing lowers clustered harm and future-utility regret without collapsing to reject-all.

## Authoritative protocol

[`EXP-006 Implementation Brief.md`](../EXP-006%20Implementation%20Brief.md), revision v2.

Key safeguards:

- Stage-0 predicted depth/pose/confidence health gate before atom training.
- K=5 utility-conditioned Stage-1 training; matched episode identity is not a label.
- First-order meta-gradient with detached discrete correspondence, inlier, Sim(3), and neighbor selection.
- Train-only neutral deadband, normalization, class weights, and grouped five-fold model selection.
- One-shot official validation; the exposed test split is rejected by the CLI.
- Undirected scene-pair/overlap-component grouped bootstrap inference.

## Data

- Manifest: `revisit3d/manifests/nuscenes_revisit_dev.json`
- Train: 20 directional episodes / 10 undirected pairs.
- Validation: 14 directional episodes / 7 undirected pairs.
- Test: 6 exposed directional episodes / 3 pairs; closed and prohibited.

## Planned configuration and outputs

- Config: `configs/EXP-006_atom_utility.yaml` (not created yet)
- Geometry checkpoint: `revisit3d/checkpoints/exp006_geometry_bootstrap.pt` (not created yet; excluded from Git)
- Compact results: `revisit3d/results/EXP-006/` (not created yet)

## Planned primary metrics

- normalized future-loss change versus current-only TTT
- grouped-bootstrap confidence interval
- cluster and directional harm rates
- future-utility regret and accept/reject rate
- alignment validity, residual, inlier ratio, and transport retention
- runtime and peak GPU memory

## Result

Pending. Do not add conclusions until Stage 0 and the one-shot validation protocol have been completed.
