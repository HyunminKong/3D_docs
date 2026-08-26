# EXP-057 — Explicit Missing-Surface Oracle

Status: Completed; all registered gates passed
Purpose: Test whether explicit past surface evidence contains geometry that
recurrent TTT3R plus repeated current optimization cannot recover

## Data and intervention

Reuse the 16 exposed EXP-052 anchors: four from each train scene `pumpkin`,
`heads`, `chess`, and `stairs`. Each anchor has four frames. Process the first
three RGB frames cleanly, then evaluate two branches from the identical
pre-target recurrent state:

- `clean`: unchanged fourth RGB;
- `erased`: set the central rectangle covering half the image width and half
  the image height to zero in normalized model-input space.

Only the erased branch is adapted. The central area is 25% of image pixels and
is fixed before execution.

## Evaluation support

Forward-warp the immediately previous frame's ground-truth depth through its
ground-truth camera pose into the target camera with nearest-pixel z-buffering.
Evaluate only central erased pixels with valid target depth and warped past
depth agreeing within 5 cm. This defines a static surface that was observed in
the past but removed from the current RGB. Require at least 2,048 supported
pixels per anchor.

All errors are median-scale-aligned relative 3D point EPE in the target camera.
Current prediction scale is estimated only from valid non-erased target pixels.

## Policies

1. clean official TTT3R target;
2. erased official TTT3R target;
3. erased target plus one generic 8-D local-code step;
4. erased target plus two equal normalized local-code steps;
5. `GT-past fusion`: replace supported pixels with the warped past GT surface;
6. `predicted-past fusion`: warp the previous clean TTT3R self-view point map
   using offline GT pose and a previous-view GT median scale, then replace
   supported pixels where predicted evidence exists;
7. `shuffled-past fusion`: deterministically permute the same predicted depths
   across supported pixels before fusion;
8. per-pixel best of second-current and predicted-past, reported only as
   diagnostic headroom.

GT pose, scale, visibility, and per-pixel best selection are offline oracles.
No policy is deployable and no model parameter or memory component is fit.

## Registered gates

- exact 16 anchors/four scenes, finite metrics, at least 2,048 supported pixels
  per anchor, predicted-past coverage at least 50%, and no validation/terminal
  access;
- erased official TTT3R is worse than clean in every scene;
- GT-past fusion beats second-current TTT in every scene with a positive paired
  anchor-bootstrap 95% interval;
- predicted-past fusion beats second-current TTT and shuffled-past fusion in
  every scene with positive paired intervals;
- predicted-past fusion harm relative to second-current is at most 25%.

Passing supports H17's oracle premise and authorizes design of one compact
visibility-addressed evidence mechanism. Failure stops architecture work and
distinguishes an absent information advantage from inadequate predicted past
surface quality.

## Artifacts

- Config: `configs/EXP-057_explicit_missing_surface_oracle_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp057_explicit_missing_surface_oracle.py`
- Result: `revisit3d/results/EXP-057/explicit_missing_surface_oracle_v10.json`

## Result

All 16 anchors completed with 26,413--46,544 supported erased pixels, exact
zero-code parity, and 59.7--98.8% predicted-past coverage (89.5% mean). Erasure
increased current error by `0.429`, CI `[0.312, 0.546]`, and the two local-code
steps were effectively unable to repair it.

GT-past fusion improved over second-current by `0.480`, CI
`[0.364, 0.596]`. More importantly, the frozen TTT3R predicted-past surface
improved by `0.408`, CI `[0.292, 0.525]`, was positive in every scene and every
anchor, and caused 0% harm. Correct spatial addressing beat a permutation of
the same predicted payload by `0.129`, CI `[0.091, 0.174]`, again in every
scene.

H17's oracle premise is supported: explicit past surface evidence contains
information that recurrent TTT3R plus repeated current local optimization does
not recover. The result is not deployable because GT pose, scale, visibility,
and the synthetic erasure mask are used. D143 authorizes only a no-fit removal
of the first three oracle dependencies in EXP-058.
