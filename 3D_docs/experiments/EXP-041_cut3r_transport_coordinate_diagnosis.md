# EXP-041 — CUT3R Transport/Coordinate Diagnosis

Status: Completed; no raw carrier was eligible
Purpose: Train-only failure decomposition; no fitting

## Question

Did EXP-040 fail because nearest predicted 3D is the wrong carrier, or because
the raw source update is incompatible with the target update under every simple
carrier?

## Fixed comparison

On the exact same 32 train pairs and fixed `0.001` current/source steps, add to
the target current code one of:

- untransported source code at the same patch index;
- soft cosine transport using frozen CUT3R encoder tokens at temperature 0.07,
  inherited from the archived visual-transport implementation;
- nearest predicted canonical-3D transport; or
- a deterministic spatial shuffle of the visual transport.

For each, measure scene-balanced future consistency, harm fraction, and cosine
agreement with the target current descent code. No parameter is fit and no
validation/terminal image is opened.

## Registered decision

A raw carrier is eligible only when it has positive mean gain over current TTT,
positive code agreement, and beats visual spatial shuffle. Among eligible
carriers, choose the lowest mean loss. If none is eligible, raw adaptation
experience is not reusable on this carrier; the result cannot authorize an
address or memory bank.

## Artifacts

- Config: `configs/EXP-041_cut3r_transport_coordinate_diagnosis_v10.yaml`
- Script: `revisit3d/scripts/diagnose_exp041_cut3r_transport_coordinate.py`
- Result: `revisit3d/results/EXP-041/cut3r_transport_coordinate_diagnosis_v10.json`

## Result

| Carrier | Gain over current | Code agreement | Harm fraction |
| --- | ---: | ---: | ---: |
| untransported | `-4.60e-6` | `-0.0527` | 50.00% |
| visual | `-8.77e-6` | `-0.0863` | 53.13% |
| canonical 3D | `-1.76e-5` | `-0.0742` | 53.13% |
| visual spatial shuffle | `-5.46e-6` | `-0.0082` | 62.50% |

Every raw carrier worsened the scene-balanced current-TTT loss and every
source/target code agreement was negative. No carrier met the registered
decision rule; the result is `no_raw_carrier`.

## Conclusion

EXP-040 was not merely a nearest-3D correspondence failure. In a generic
orthonormal 8-D coordinate, source and target descent codes are systematically
incompatible even for pose-defined revisits. The paper may not claim that raw
TTT directions are naturally reusable and may not proceed directly to an
address or bank.

The minimal surviving hypothesis is narrower: offline training may learn the
single shared `8 -> 768` plasticity basis so that the same one online loss
produces current-useful and revisit-compatible codes. This changes no online
module or loss. It must first pass a train-internal scene-disjoint oracle test
before validation is opened.
