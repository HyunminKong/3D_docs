# EXP-021 — Independent Benchmark Feasibility

Status: Completed; terminal manifest frozen without sensor/model access

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

## Stage 0 result

All gates passed without decoding RGB, LiDAR, or model output. All 150 official
test scenes were eligible. The graph contains 107 undirected edges over 96
scenes and 29 connected components in three locations. All 6,008 CAM_FRONT and
6,008 LIDAR_TOP keyframe paths exist. Scene-token overlap with trainval and
scene-name overlap with every previous manifest are both zero.

## Stage 1 registration

Because one Boston component contains 54 of 107 edges, splitting these 29
components into both a statistically broad development and final subset would
needlessly weaken one side. The existing 25-component training benchmark will
remain the only model-development source, using source-safe component OOF. All
29 untouched official-test components are reserved for a single terminal test.

Stage 1 freezes both directions of all 107 edges with eight context and four
disjoint query frames. It requires at least 200 directional episodes, 90 scenes,
25 components, and all three locations. This remains metadata-only and does not
authorize pixel, LiDAR, feature, or model access.

Additional Stage 1 files:

- Config: `configs/EXP-021_terminal_manifest_v11.yaml`
- Builder: `revisit3d/scripts/build_exp021_terminal_manifest.py`
- Manifest: `revisit3d/manifests/nuscenes_official_test_revisit_exp021_v11.json`
- Result: `revisit3d/results/EXP-021/stage1_terminal_manifest_v11.json`

## Stage 1 result

All registered health checks passed. The immutable terminal manifest contains
214 directional episodes, 96 scenes, 29 connected components, and three
locations. Its SHA-256 is
`22f7ec04caf83ead7efef828dab3231c7919757d13f88509b66ea0257ea95d61`.
No RGB, LiDAR, feature, or model output was accessed. The manifest is closed
until a replacement method, artifact hash, metrics, and terminal gates are
pre-registered.
