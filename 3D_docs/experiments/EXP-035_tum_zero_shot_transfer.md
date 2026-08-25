# EXP-035 — Frozen nuScenes→TUM Zero-Shot Transfer

Status: Registered before TUM sensor decoding or model access

## Question

Does the exact nuScenes-selected local adaptation-memory policy retain a useful
effect on indoor TUM RGB-D streams without any TUM fitting or selection?

## Frozen protocol

- immutable EXP-028 atom and EXP-029 factorized address hashes;
- immutable one `track3D` step at 0.0125, 8-D code, visual transport, residual
  0.10, top-1 semantic-zero acceptance;
- deterministic reservoir-64 per sequence, predict before write;
- frozen FastVGGT and the existing custom base geometry head;
- no TUM PCA, refit, calibration, threshold, loss, seed, or capacity change.

Stage 1 decodes only RGB and caches foundation geometry/tracks for all 223
contexts and their read-only query views. Stage 2 computes every online
decision before decoding query depth. Dense TUM depth is used only for SILog,
median-aligned AbsRel, and 3D EPE evaluation.

Controls are current-only, same-bank random expectation, and appearance
retrieval under identical semantic-zero acceptance. Results are sequence
balanced and also reported per sequence because 98 of 111 targets come from
Freiburg2-xyz.

## Registered descriptive gate

At least 100 targets and all three sequences must be evaluable. Full memory
must improve the sequence-balanced mean of all three primary errors over each
control. Sequence-bootstrap intervals are reported but are not a success gate
because three groups cannot provide paper-level inference. Any failure is
reported without TUM-side repair or rerun.

## Files

- Config: `configs/EXP-035_tum_zero_shot_transfer_v10.yaml`
- Cache script: `revisit3d/scripts/cache_exp035_tum_geometry.py`
- Evaluation script: `revisit3d/scripts/evaluate_exp035_tum_zero_shot.py`
- Result: `revisit3d/results/EXP-035/stage2_tum_zero_shot_transfer_v10.json`
