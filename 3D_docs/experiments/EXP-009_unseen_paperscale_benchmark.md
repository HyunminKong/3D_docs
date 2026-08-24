# EXP-009 — Fully Unseen Paper-Scale Revisit Benchmark

Status: **Stage 0 passed; Stage 1 manifest freeze registered.**

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

## Stage-0 result

After blacklisting 130 previous scenes, 719 unseen scenes remained. Metadata-only matching found 1,368 undirected overlap edges over 636 scenes and 65 connected components. The fixed split produced:

- train: 1,134 undirected / 2,268 directional episodes, 454 scenes;
- validation: 117 / 234 episodes, 86 scenes;
- locked test: 117 / 234 episodes, 96 scenes.

Scene and component intersections are zero. Validation and test each contain components from all four official locations. The largest 304-scene/936-edge component is isolated in train.

## Stage-1 manifest freeze

Generate both directions of every fixed edge with eight context and four disjoint query frames. A/A′ use a 15%-length window around the closest pose anchors; B uses the opposite temporal end of the source traversal. Before writing the manifest, assert unique episode IDs, valid/disjoint indices, blacklist exclusion, minimum split size, four-location holdout coverage, and zero scene/component intersections. No image or model output is read.

## Outputs

- Config: `configs/EXP-009_unseen_benchmark_inventory_v10.yaml`
- Script: `revisit3d/scripts/build_exp009_unseen_inventory.py`
- Full inventory: `revisit3d/cache/EXP-009/unseen_overlap_inventory_v10.json`
- Summary: `revisit3d/results/EXP-009/stage0_unseen_overlap_inventory_v10.json`
- Manifest config: `configs/EXP-009_unseen_manifest_v11.yaml`
- Frozen manifest: `revisit3d/manifests/nuscenes_revisit_unseen_exp009_v11.json`
