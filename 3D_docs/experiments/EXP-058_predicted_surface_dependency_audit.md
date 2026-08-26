# EXP-058 — Predicted-Surface Dependency Audit

Status: Registered; train-only no-fit dependency decomposition
Purpose: Determine whether EXP-057's explicit-surface advantage survives after
removing GT pose, past metric scale, and GT visibility from fusion

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
