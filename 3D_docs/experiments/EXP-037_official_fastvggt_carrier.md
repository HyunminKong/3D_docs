# EXP-037 — Official FastVGGT Geometry-Carrier Diagnostic

Status: Completed; registered carrier gate failed
Purpose: Exposed engineering selection; never a final paper test

## Question

Can the official frozen FastVGGT depth head remove the absolute-geometry
bottleneck while preserving the 224 x 224 token interface already used by the
local plasticity method?

## Why this experiment comes first

EXP-036 showed that the custom geometry head is 1.8--3.0 times worse than
TTT3R on matched TUM targets. Replacing the entire state model immediately
would introduce a large architectural change and overlap more directly with
TTT3R. The official FastVGGT head is therefore the minimal carrier candidate:
it uses the same foundation checkpoint and query token frame as the archived
mechanism.

## Frozen protocol

- Use the already exposed EXP-034 TUM manifest and all 111 registered revisit
  targets across three sequences.
- Run the official frozen FastVGGT depth head at 224 x 224 on query views only.
- Query RGB is required for prediction, but query frames do not update any
  streaming or adaptation state.
- Decode dense TUM depth only after prediction and use the exact EXP-035 metric
  definitions and sequence-balanced aggregation.
- Perform no TUM training, calibration, threshold search, scale fitting beyond
  the already registered per-view median alignment metric, or memory fitting.

## Registered carrier decision

The official head is a viable minimal carrier only if:

1. all 111 targets and all three sequences produce finite metrics;
2. it improves SILog, aligned AbsRel, and 3D EPE over the archived custom-head
   full method; and
3. every primary error is at most 1.25 times the EXP-036 TTT3R error.

If all conditions pass, the next branch integrates the plasticity residual and
utility address with the official FastVGGT geometry output. If any condition
fails, the next branch moves to a CUT3R/TTT3R-class recurrent carrier. The
threshold is a carrier-selection rule, not a paper claim.

## Artifacts

- Config: `configs/EXP-037_official_fastvggt_carrier_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp037_official_fastvggt_carrier.py`
- Result: `revisit3d/results/EXP-037/official_fastvggt_carrier_v10.json`

## Result

The official head evaluated all 111 targets and all three sequences. Its
sequence-balanced means were:

| Carrier | SILog | aligned AbsRel | 3D EPE (m) |
| --- | ---: | ---: | ---: |
| Archived custom head + full memory | 28.4625 | 0.230136 | 0.458924 |
| Official FastVGGT head | 18.1625 | 0.111742 | 0.296975 |
| EXP-036 TTT3R | 15.7271 | 0.078073 | 0.224615 |

Official FastVGGT reduced the custom system's errors by 36.19%, 51.45%, and
35.29%, respectively. Relative to TTT3R, however, its error ratios were 1.155,
1.431, and 1.322. AbsRel and EPE exceeded the registered 1.25 limit.

The run used 15.40 seconds for 111 targets (0.139 seconds/target) and peaked at
5.05 GB allocated GPU memory. The result artifact hash is
`fc23115d0cab8ff646bc0a796c516c5d6a8156657f7b7834f54caaa5b31f878f`.

## Conclusion

The official FastVGGT head proves that most of the archived model's absolute
error came from the custom decoder, but it is not close enough to the strongest
matched recurrent carrier under the preregistered rule. The minimal official
FastVGGT integration is rejected. The next branch must integrate the local
adaptation-memory principle with a CUT3R/TTT3R-class recurrent carrier.

This is exposed engineering-selection evidence only. It must not be reported
as held-out paper evidence or used to tune the eventual integrated model.
