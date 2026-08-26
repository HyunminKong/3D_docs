# EXP-065 — Calibration-Shock State-Poisoning Anatomy

## Status

Corrected v1.1 completed; gate failed. No model was fit.

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

## Result

All 16 contexts, 96 condition paths, and both deterministic controls completed.
Clean replay and `clean_skip`/`zoom_skip` state equivalence were exact. The mean
post-recovery clean-query EPE was `0.069581`. The zoom difference-in-differences
penalty was `0.001030`, only 1.48% of clean EPE versus the frozen 5% minimum.
It was positive in 11/16 contexts (`68.75%`) versus the required 75%, and its
stratified bootstrap interval `[-0.000351, 0.002366]` crossed zero.

The mean penalty was positive in all four scenes, and the immediate penalty had
a positive CI `[0.000449, 0.003619]`. This shows that zoomed input is initially
more disruptive. The persistent attribution is not secure: zoom exceeded the
resampling control by only `0.000698` with CI
`[-0.000646, 0.001937]`, and exceeded the missing-periphery control by
`0.001029` with CI `[-0.000714, 0.002848]`. The latter comparison was negative
in `stairs` (`-0.002285`).

Frozen gates passed only exact coverage/replay/skip equivalence, all-scene mean
persistent excess, all-scene mean zoom-over-resampling, and all-scene immediate
plus persistent signs. The magnitude, context-frequency, confidence-bound, and
missing-periphery attribution gates failed. Overall status: failed.

## Interpretation and conclusion

A 4/3 focal/FoV shock has a measurable immediate effect and a small positive
mean residual after one clean recovery update. It is too small, heterogeneous,
and confounded with visible-support loss to carry a compact top-tier paper on
this frozen carrier. H21 is rejected as the central method premise. The project
will not increase zoom strength, choose favorable contexts, or add a
calibration gate/encoder on this evidence.

## Verification

```bash
PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python \
  revisit3d/scripts/prepare_exp065_selected_train_depth.py \
  --config configs/EXP-065_calibration_shock_anatomy_v10.yaml \
  --confirm-selected-train-depth-registration

PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python \
  revisit3d/scripts/evaluate_exp065_calibration_shock_anatomy.py \
  --config configs/EXP-065_calibration_shock_anatomy_v10.yaml \
  --confirm-train-rgbd-calibration-anatomy
```

- peak allocated GPU memory: `4,955,941,376` bytes;
- validation/terminal accessed: no/no;
- depth-preparation SHA-256:
  `1216e0e82b8121a81118623adf7edbe7d8b5e5352ece921fdac5d3821112cdec`;
- result SHA-256:
  `037a5f4e3b2f35114498311c595e75b778eae6294cadc5813b7e98907c6234ac`.

## Artifacts

- Config: `configs/EXP-065_calibration_shock_anatomy_v10.yaml`
- Depth preparation:
  `revisit3d/results/EXP-065/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-065/calibration_shock_anatomy_v10.json`
