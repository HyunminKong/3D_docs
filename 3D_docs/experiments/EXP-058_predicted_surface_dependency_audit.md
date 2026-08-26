# EXP-058 — Predicted-Surface Dependency Audit

Status: Completed; all functional gates passed, reproduction guard failed
Purpose: Determine whether EXP-057's explicit-surface advantage survives after
removing GT pose, past metric scale, and GT visibility from fusion

Protocol revision v1.1 changes only import order after the registered v1.0
runner stopped before model construction or data access on an external dust3r
camera/head circular import. The carrier now initializes before importing the
pose-decoding utility; method, data, controls, and gates are unchanged.

## Protocol

Reuse the exact EXP-057 16 controlled-erasure anchors, carrier, mask, current
steps, metric, and evaluation support. The evaluation support may still use GT
to score the same previously-visible static surfaces, but the fusion policy may
not access GT pose, depth, scale, or visibility.

Decode frozen TTT3R source and erased-target pose encodings into camera-to-
shared-frame transforms. Move the stored source self-view pointmap into the
target camera using those predicted poses, z-buffer it with the known calibrated
target intrinsics, and fuse valid projected depths only inside the known
synthetic erasure mask. Source and target pointmaps remain in the native
predicted scale. Apply the same evaluation-only median scale estimated outside
the erasure to the complete fused target pointmap.

Compare:

1. erased official TTT3R;
2. one and two current local-code steps;
3. EXP-057 GT-aligned predicted-past oracle reference;
4. predicted-only past-surface fusion;
5. a within-erasure spatial permutation of the identical predicted-only depth
   payload;
6. per-pixel best predicted-only/current diagnostic.

No parameter, threshold, visibility model, or memory bank is fit. Known
intrinsics and the synthetic erasure mask are the only fusion-side metadata.

## Registered gates

- exact 16-anchor/four-scene coverage, finite values, EXP-057 second-current
  reproduction within `1e-5`, at least 50% predicted-only coverage, and no
  validation/terminal access;
- predicted-only fusion beats second-current TTT and spatial shuffle in every
  scene with positive paired anchor-bootstrap intervals;
- predicted-only fusion harms at most 25% of anchors;
- predicted-only mean gain retains at least 50% of EXP-057's immutable
  GT-aligned predicted-past gain.

Passing authorizes design of one minimal observable visibility/address rule.
Failure localizes the blocker to pose/scale/visibility deployment and prevents
learned fusion or bank construction.

## Artifacts

- Config: `configs/EXP-058_predicted_surface_dependency_audit_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp058_predicted_surface_dependency_audit.py`
- Result: `revisit3d/results/EXP-058/predicted_surface_dependency_audit_v10.json`

Result SHA-256:
`bae3edafd5dd72b664debd486f5b1827eec6af843c2654d29927a40d800cbac7`.

## Result

Predicted-pose/native-scale fusion achieved 89.57% mean evaluation coverage.
It improved relative 3D EPE over second-current TTT by `0.3940`, with paired
anchor-bootstrap CI `[0.2866, 0.5054]`, and beat the identical-payload spatial
shuffle by `0.1188`, CI `[0.0815, 0.1617]`. Both gains were positive in every
scene, observed harm was 0%, and the mean gain retained 97.59% of EXP-057's
GT-aligned predicted-surface oracle gain.

The registered gate nevertheless failed. Four of 16 repeated second-current
errors differed from EXP-057 by more than the fixed `1e-5` tolerance; the
maximum was `1.96695e-5`. No tolerance, result, or gate was changed and the
experiment was not rerun.

## Interpretation

The dependency-removal mechanism itself passes every registered functional
comparison by a large margin: GT relative pose, past scale, and GT visibility
are not needed for the controlled fusion result. EXP-059 separately establishes
that row identities and evaluation support match and that the pre-adaptation
erased baseline is bit-exact; the small guard miss appears only after repeated
local-code differentiation. EXP-058 therefore remains a literal failed gate
but supplies qualified positive dependency evidence. It does not establish a
deployable natural visibility rule, a multi-frame address, or a novel generic
geometry memory.
