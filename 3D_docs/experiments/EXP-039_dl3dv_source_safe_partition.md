# EXP-039 — DL3DV Source-Safe Revisit Partition

Status: v1.0 preserved with one missing-path failure; corrected v1.1 passed all gates
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

- Config v1.0: `configs/EXP-039_dl3dv_source_safe_partition_v10.yaml`
- Corrected config v1.1: `configs/EXP-039_dl3dv_source_safe_partition_v11.yaml`
- Script: `revisit3d/scripts/build_exp039_dl3dv_partition.py`
- Result v1.0: `revisit3d/results/EXP-039/dl3dv_partition_audit_v10.json`
- Corrected result v1.1: `revisit3d/results/EXP-039/dl3dv_partition_audit_v11.json`
- Manifests: `revisit3d/manifests/dl3dv_recurrent_revisit_*_exp039_v1*.json`

## v1.0 path audit

Every metadata, eligibility, split, pair-count, disjointness, and no-access
check passed. The sole failure was one missing local file,
`frame_00286.png`, referenced as `source-1` in one terminal pair. Neighboring
files and 354 other frames exist, so this is a single-file dataset inventory
gap rather than a model or revisit failure.

Version 1.1 excludes any pair with a missing required RGB path before the same
deterministic maximum-16 subsampling. It changes no pose threshold, eligibility
minimum, split seed/size, or success gate. The v1.0 result and manifests remain
preserved.

## Corrected v1.1 result

All gates passed. Ninety-one eligible scenes were split without overlap into:

| Role | Scenes | Pairs | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| train | 63 | 982 | `539722145fb4babc9a8526cdbd56e27acb77f5e32a4cc91ab6e165e5ba5f6f05` |
| validation | 14 | 213 | `79935771c8d882ee0ee73e8d3b5556de48f1afe516e255474b63e9bd53964ee4` |
| terminal | 14 | 224 | `49e6c389048fb41194970538a021f5345ae3006ac306d7bbc70fe62b591b89d6` |

No image was decoded, no geometry/depth label was accessed, and no model output
was produced. The terminal manifest is locked and cannot enter basis, step-size,
address, or policy selection.

## Conclusion

The integrated branch now has a genuinely new source-safe development path.
EXP-040 may open only a registered train subset to test current-code and oracle
transport utility. Validation remains one-shot and terminal remains closed.
