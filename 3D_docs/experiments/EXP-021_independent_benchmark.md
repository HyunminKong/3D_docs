# EXP-021 — Independent Benchmark Feasibility

Status: Registered before metadata audit

## Question

Do untouched nuScenes official-test scenes contain enough physical revisits,
independent overlap components, camera frames, and sparse LiDAR files to support
a new development/final-test protocol after EXP-020 exposed validation?

## Stage 0 protocol

Read only official scene, sample, calibration, ego-pose, timestamp, log, and
sample-data metadata from `v1.0-test`. Do not decode RGB, LiDAR, or model output.
Build same-location cross-scene edges when CAM_FRONT camera centers approach
within 2.0 m. Connected overlap components are indivisible future split units.
Audit official trainval scene-token disjointness, previous-manifest scene-name
disjointness, and camera/LiDAR keyframe file existence.

The feasibility gate requires at least 140 eligible scenes, 50 overlap scenes,
40 undirected edges, 10 independent components, three locations, zero previous
scene overlap, and 100% keyframe-file existence. Passing authorizes a
metadata-only development/final-test split design; it does not authorize pixel,
LiDAR, feature, or model access.

## Files

- Config: `configs/EXP-021_independent_test_inventory_v10.yaml`
- Auditor: `revisit3d/scripts/audit_exp021_independent_test_inventory.py`
- Inventory cache: `revisit3d/cache/EXP-021/independent_test_inventory_v10.json`
- Summary: `revisit3d/results/EXP-021/stage0_independent_test_inventory_v10.json`
