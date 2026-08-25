# EXP-038 — Recurrent Carrier Plasticity Interface Audit

Status: v1.0 preserved with implementation failure; corrected v1.1 passed all gates
Purpose: Exposed engineering/interface evidence; no model fitting

## Question

Can one local 8-D test-time code act on an official frozen CUT3R geometry
carrier while satisfying all three method requirements?

1. zero code preserves the competitive base prediction;
2. one online 3D consistency loss differentiates into the code; and
3. the local code can be transported between frames in the carrier's predicted
   canonical 3D coordinates.

## Minimal interface

CUT3R, rather than TTT3R's attention update, is the base recurrent rule. The
official encoder, recurrent state, pose memory, DPT geometry head, and weights
are frozen. A shared `8 -> 768` linear basis maps one code per decoder patch to
a residual on only the last image-token level before the official DPT head.
The pose token is untouched. There is no replacement geometry head, router,
risk module, or auxiliary loss.

The online diagnostic loss is one symmetric nearest-neighbor consistency loss
between adjacent predicted point sets in CUT3R's common `other_view` frame.
The audit takes one normalized infinitesimal step only to establish that the
interface has a useful descent direction; it does not select a paper learning
rate or train the basis.

## Registered checks

- custom step interface matches native CUT3R within `1e-5` maximum absolute
  geometry-output error on three chronological RGB frames;
- zero local code matches the custom base path within `1e-6`;
- the online loss produces a finite code-gradient norm at least `1e-8`;
- the fixed `0.001` normalized diagnostic step lowers that loss and changes
  geometry;
- self-to-self nearest-neighbor 3D transport reconstructs the code within
  `1e-6` RMSE; and
- adjacent-frame transport is finite.

All checks must pass before any integrated basis training or utility-memory
fitting. TUM depth and query labels are not opened.

## Artifacts

- Config v1.0: `configs/EXP-038_recurrent_carrier_interface_v10.yaml`
- Corrected config v1.1: `configs/EXP-038_recurrent_carrier_interface_v11.yaml`
- Carrier: `revisit3d/backbones/recurrent_carrier.py`
- Script: `revisit3d/scripts/audit_exp038_recurrent_carrier_interface.py`
- Result v1.0: `revisit3d/results/EXP-038/recurrent_carrier_interface_v10.json`
- Corrected result v1.1: `revisit3d/results/EXP-038/recurrent_carrier_interface_v11.json`

## v1.0 implementation audit

The first execution is preserved as a failed result. Zero-code parity, finite
code gradients, nonzero geometry response, and finite cross-view transport
passed. Three checks failed for two localized implementation reasons:

- the custom rollout requested `return_attn=false`, causing the external model
  to switch to SDPA while native lighter inference uses explicit attention;
  the two valid kernels differed by up to 0.0176 numerically; and
- `torch.cdist` suffered cancellation on large canonical coordinates, giving
  nonzero self-distances up to 0.1265. This invalidated identity transport and
  the diagnostic descent calculation.

Version 1.1 changes only those implementations: it follows native explicit
attention and computes Euclidean distances from direct coordinate differences.
The probe frames, 8-D basis, `0.001` step, thresholds, and all success checks
remain unchanged. This is a correction, not a selected method variant.

## Corrected v1.1 result

All registered checks passed:

- native CUT3R versus carrier step maximum absolute error: `0.0`;
- zero-code maximum absolute error: `0.0`;
- online code-gradient norm: `9.224e-4`;
- adjacent-point consistency: `0.232733 -> 0.232658` after the fixed
  diagnostic step;
- nonzero-code maximum geometry change: `0.01977`;
- identity 3D transport code RMSE and maximum distance: `0.0`; and
- adjacent-frame mean 3D transport distance: `0.09502`, finite.

One float32 atom stores 768 x 8 values, or 24 KiB. The single shared residual
basis contains 6,144 parameters. No fitting or query/depth access occurred.

## Conclusion

The selected interface is technically valid: it is an exact zero-residual
extension of official CUT3R, supplies a differentiable one-loss TTT coordinate,
and supports explicit transport in predicted canonical 3D. This authorizes a
new source-safe premise experiment for utility of current and revisited codes;
it does not yet show that the code improves held-out reconstruction.
