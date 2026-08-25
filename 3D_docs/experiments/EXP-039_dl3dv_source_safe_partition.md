# EXP-039 — DL3DV Source-Safe Revisit Partition

Status: Registered before execution
Purpose: Metadata-only partition; no pixels, labels, or model outputs

## Question

Can the locally available, previously unused DL3DV benchmark supply enough
scene-disjoint physical revisits to train and validate the integrated CUT3R
local-code mechanism without reusing exposed nuScenes/TUM selection data?

## Fixed revisit definition

For each scene, camera centers are normalized by the median nonzero adjacent
camera step. A target is a revisit when at least one source:

- is at least 30 frames earlier;
- lies within five median camera steps; and
- differs by at most 30 degrees in camera orientation.

The closest eligible source is paired with each target. A usable scene has at
least eight pairs; at most 16 evenly spaced pairs are retained. Each pair stores
`source-1, source, target-1, target`, enabling the one adjacent-frame online
loss at source and an offline future-utility measurement at target.

## Fixed split

Eligible scene IDs are ordered by SHA-256 of `scene_id:3900010` and assigned
without overlap:

- 63 train scenes;
- 14 validation scenes; and
- 14 terminal scenes.

The terminal manifest is created and hashed before any RGB decode, carrier
output, basis fitting, utility fitting, or target metric access. It may not be
opened during model selection.

## Gate

The audit passes only with exactly the registered 141-file metadata inventory,
91 eligible scenes, exact split sizes, at least 504/112/112 pairs, disjoint
scene IDs, existing RGB paths, and zero sensor/model access.

DL3DV supplies camera trajectories but no dense depth in this local release.
It can support source-safe basis/address development and RGB-only future
consistency, but it cannot be the sole terminal evidence for absolute depth or
3D EPE. A separate unused RGB-D benchmark remains mandatory.

## Artifacts

- Config: `configs/EXP-039_dl3dv_source_safe_partition_v10.yaml`
- Script: `revisit3d/scripts/build_exp039_dl3dv_partition.py`
- Result: `revisit3d/results/EXP-039/dl3dv_partition_audit_v10.json`
- Manifests: `revisit3d/manifests/dl3dv_recurrent_revisit_*_exp039_v10.json`
