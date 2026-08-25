# EXP-038 — Recurrent Carrier Plasticity Interface Audit

Status: Registered before execution
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

- Config: `configs/EXP-038_recurrent_carrier_interface_v10.yaml`
- Carrier: `revisit3d/backbones/recurrent_carrier.py`
- Script: `revisit3d/scripts/audit_exp038_recurrent_carrier_interface.py`
- Result: `revisit3d/results/EXP-038/recurrent_carrier_interface_v10.json`
