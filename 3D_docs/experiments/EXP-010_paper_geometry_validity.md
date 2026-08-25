# EXP-010 — Paper-Critical Absolute Geometry Validity

Status: Registered before execution

## Question

Does the locked EXP-009 utility-addressed plasticity model improve actual sparse-LiDAR depth/3D geometry, rather than only the self-supervised future-loss proxy?

## Boundary

EXP-009 weights, address, router, threshold, K=5, residual 0.10, and capacity 64 remain byte-frozen. EXP-010 is a post-lock secondary-endpoint audit: it was registered after observing the EXP-009 proxy endpoint, so it is not presented as a preregistered confirmatory test. It may narrow the claim but cannot tune or alter the method.

Query LiDAR is evaluation-only. It never enters TTT, addressing, transport, routing, or bank retention.

## Stage A — Decisive metric bridge

Replay the same causal reservoir-64 stream and recover predictions for:

1. frozen base geometry;
2. current-only one-step TTT;
3. the locked full memory model.

Project nuScenes `LIDAR_TOP` points into each held-out CAM_FRONT query frame and aggregate the nearest point per 8x8 prediction cell. Evaluate only finite points between 1 m and 80 m.

Primary metrics:

- scale-invariant log error (SILog);
- per-view median-aligned AbsRel;
- per-view median-aligned 3D same-ray endpoint error.

Secondary metrics are aligned RMSE and delta-1 accuracy. Aggregation is first per target, then by physical-overlap component. Paired component bootstrap uses 10,000 samples.

## Registered Stage-A gate

Stage A passes only if:

1. at least 20 components and 90 targets have valid LiDAR coverage;
2. full memory is no worse than current-only TTT in mean SILog, aligned AbsRel, and aligned 3D endpoint error;
3. at least one of those three improvements has a paired component-bootstrap 95% interval strictly above zero;
4. the fraction where full memory degrades current-only AbsRel by more than 1% is no greater than the fraction where current-only TTT degrades the frozen base by more than 1%.

Failure stops architecture expansion and narrows the paper claim. Success authorizes Stage B controls: random address, address-only, full router, FIFO/unbounded, efficiency, and minimal-loss refitting on train/validation only.

## Files

- Config: `configs/EXP-010_paper_geometry_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp010_absolute_geometry.py`
- Result: `revisit3d/results/EXP-010/stageA_absolute_geometry_test_v10.json`
