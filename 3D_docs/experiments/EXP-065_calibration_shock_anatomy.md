# EXP-065 — Calibration-Shock State-Poisoning Anatomy

## Status

Corrected v1.1 preregistration; not yet run. The first launch stopped before
loading any context or producing model output because selected projected-depth
files had not been materialized.

## Question

Does one physically valid focal/FoV change produce persistent recurrent-state
damage on later clean 3D queries beyond resampling and missing-periphery image
controls?

## Protocol

- Use 16 fresh train-only 7Scenes contexts: four fixed contexts from each of
  `chess`, `heads`, `pumpkin`, and `stairs`. Validation and terminal roles stay
  closed.
- Each context contains two clean history frames, one intervention frame, one
  clean recovery frame, and one identical clean `update=false` query.
- The fixed zoom intervention center-crops the middle 3/4 of both image axes and
  bilinearly resizes it to the original tensor size. This is exactly a 4/3
  pinhole focal/FoV change at the same optical center.
- Evaluate six independently replayed paths:
  `clean_write`, `zoom_write`, `clean_skip`, `zoom_skip`,
  `resample_write`, and `periphery_mask_write`.
- `resample_write` downsamples the full-FoV image to 3/4 size and upsamples it,
  matching interpolation/bandwidth change without changing the FOV.
- `periphery_mask_write` replaces the same outer 43.75% image area removed by
  zoom with the per-channel image mean, preserving central pixel coordinates
  and isolating missing peripheral evidence.
- Before the clean recovery write, query the recovery frame with
  `update=false` for the immediate effect. Then write that same clean frame and
  query the fifth clean frame with `update=false` for the persistent effect.
- RGB alone enters frozen TTT3R. RGB-D and calibrated intrinsics are offline
  metric labels only. Every query pointmap is independently median-depth aligned
  before relative 3D EPE.

Define the persistent excess write penalty

```text
[(E_zoom_write - E_zoom_skip) - (E_clean_write - E_clean_skip)].
```

This difference-in-differences removes the effect of merely omitting the third
state update. The `clean_skip` and `zoom_skip` states must be identical because
the intervention is non-writing in both paths.

## Frozen success gate

All must hold:

1. exact 4-scene/16-context coverage and deterministic clean replay within
   `1e-5` point difference;
2. final `clean_skip` and `zoom_skip` predictions agree within `1e-5`;
3. persistent excess write penalty is positive in every scene with a positive
   stratified context-bootstrap 95% lower bound;
4. the aggregate persistent excess is at least 5% of clean-write EPE and is
   positive in at least 75% of contexts;
5. zoom-write is worse than both resample-write and periphery-mask-write in
   every scene, with positive stratified bootstrap lower bounds for both
   comparisons;
6. the immediate zoom excess and the post-clean-recovery zoom excess are both
   positive in every scene.

Failure rejects this synthetic calibration-shock premise without changing zoom
strength, choosing contexts, or adding another corruption. Success authorizes
only a fresh real-data and method-capacity decision; it does not authorize a
camera-prior encoder, TTT loss, or validation access.

## Pre-result implementation correction

The first v1.0 launch loaded the frozen checkpoint, then failed on the first raw
context because `frame-000111.depth.proj.png` did not exist. No image context,
prediction, metric, or result artifact was produced. The underlying raw train
depth exists; 7Scenes requires a deterministic registration into the RGB frame.
Revision v1.1 adds the same selected-frame registration used by EXP-061/062 and
does not change contexts, intervention, controls, metrics, or success gates.

## Artifacts

- Config: `configs/EXP-065_calibration_shock_anatomy_v10.yaml`
- Depth preparation:
  `revisit3d/results/EXP-065/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-065/calibration_shock_anatomy_v10.json`
