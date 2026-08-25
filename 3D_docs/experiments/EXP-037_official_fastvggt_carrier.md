# EXP-037 — Official FastVGGT Geometry-Carrier Diagnostic

Status: Registered before execution
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
