# EXP-009 — Fully Unseen Paper-Scale Revisit Benchmark

Status: **Stage 0 metadata inventory registered.**

## Question

Can the selected dual-address local-plasticity architecture generalize to an independently constructed, component-disjoint benchmark whose scenes were never used in EXP-001–008?

## Stage-0 boundary

Stage 0 reads nuScenes metadata only. It does not open camera images, foundation features, depth predictions, utility labels, or any existing holdout result.

- Blacklist every scene directory under the three previously converted roots and every scene named by an existing manifest.
- Keep unseen CAM_FRONT scenes with at least 200 frames.
- Compare scenes only within the same official nuScenes location.
- Create an undirected edge when any two camera centers are within 2.0 m.
- Split only by connected overlap component; no scene may cross train/validation/test.
- Use deterministic greedy balancing of undirected edge counts within each location at 70/15/15.
- Record both a full ignored inventory and a compact tracked audit.

If connected components are too large for a credible split, Stage 0 must stop and redesign the sampling unit using metadata only. It may not inspect images or model performance to repair the split.

## Outputs

- Config: `configs/EXP-009_unseen_benchmark_inventory_v10.yaml`
- Script: `revisit3d/scripts/build_exp009_unseen_inventory.py`
- Full inventory: `revisit3d/cache/EXP-009/unseen_overlap_inventory_v10.json`
- Summary: `revisit3d/results/EXP-009/stage0_unseen_overlap_inventory_v10.json`
