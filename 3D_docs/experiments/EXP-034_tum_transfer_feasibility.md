# EXP-034 — TUM Zero-Shot Transfer Feasibility Audit

Status: Registered before image/depth decoding or model access

## Question

Can the locally available TUM RGB-D data support a causal physical-revisit
stream for a no-fit nuScenes→indoor transfer test of the frozen paper model?

## Protocol

Read only `rgb.txt`, `depth.txt`, and `groundtruth.txt`; verify referenced file
existence without decoding sensors. Associate RGB, depth, and pose within 20 ms.
Create stream anchors every 30 associated frames. Each anchor uses eight context
views at frame stride three and four later read-only query views at the same
stride.

A target is a physical revisit when at least one earlier anchor is separated by
15 seconds, within 0.5 m, and within 45 degrees. All three local sequences are
reserved for a descriptive zero-shot evaluation; no TUM model fitting,
threshold selection, PCA fitting, or checkpoint selection is permitted.

## Gate

Require all three sequences, at least 200 stream contexts, at least 100 revisit
targets, complete RGB/depth/pose association for emitted views, and no image,
depth, or model decoding. Passing authorizes one frozen zero-shot EXP-035; it
does not provide paper-level independent-component inference because only three
sequences are available.

## Files

- Config: `configs/EXP-034_tum_transfer_feasibility_v10.yaml`
- Script: `revisit3d/scripts/audit_exp034_tum_transfer_feasibility.py`
- Manifest: `revisit3d/manifests/tum_zero_shot_stream_exp034_v10.json`
- Result: `revisit3d/results/EXP-034/tum_transfer_feasibility_v10.json`
