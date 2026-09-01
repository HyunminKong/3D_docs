# EXP-071 — Future-Revisit Geometry Teacher Premise

## Status

Registered; not yet executed.

## Question

After a frozen TTT3R carrier observes later frames, does rereading the same
past RGB from the final state produce a gauge-normalized 3D correction that is
more accurate than both the original causal prediction and an equal reread
from the target's own prefix state?

## Hypothesis

H27. Later multi-view evidence makes the carrier's own future-revisit output a
useful dense offline teacher for causal geometry.

## Data contract

- EXP-051 7Scenes train role only; validation and terminal roles remain closed.
- Six previously sensor-unopened train sequences:
  `pumpkin/seq-06`, `pumpkin/seq-08`, `chess/seq-05`, `chess/seq-06`,
  `stairs/seq-05`, and `stairs/seq-06`.
- Three frozen 16-frame windows per sequence starting at 64, 224, and 384.
- Fixed target offsets 3, 7, and 11 in every window: 18 contexts and 54 target
  predictions.
- RGB alone enters frozen TTT3R. Registered depth/intrinsics are offline
  evaluation labels only.

## Zero-fit protocol

1. Reset the official step-wise TTT3R recurrence at each 16-frame window.
2. At each target, save the ordinary first-pass self-view pointmap before its
   state write is available to later frames.
3. Immediately after that target is written, reread the identical target RGB
   with `update=false` from its prefix state. This is the matched self-reread
   control.
4. After all 16 RGB frames have been written, reread every target independently
   from the same final state with `update=false`. This is the future-revisit
   teacher; query order cannot alter the final state.
5. Repeat the middle target future readout once from the identical saved state
   and require numerical replay.
6. On a common finite RGB-D mask, score relative camera-frame 3D EPE after an
   independent median-depth scale removal for each output.
7. Normalize each output pointmap by its own median depth and measure cosine
   alignment between the future-minus-prefix correction and the
   target-minus-prefix metric residual.

The future teacher is offline evidence only. No model parameter, state rule,
threshold, frame, or split is fitted or selected.

## Primary metrics

- future-revisit EPE gain over prefix-revisit and over first pass;
- relative future-over-prefix gain and target win fraction;
- scene-balanced context-bootstrap interval for future-over-prefix gain;
- gauge-normalized correction/oracle-residual cosine;
- exact replay maximum absolute point difference.

## Frozen success gate

All must hold:

1. exactly 3 scenes, 6 sequences, 18 contexts, 54 targets, and 18 replay checks;
2. replay maximum absolute difference `<=1e-5`;
3. future-over-prefix EPE gain is positive in every scene and the stratified
   context-bootstrap 95% lower bound is positive;
4. aggregate relative future-over-prefix gain is at least 5%;
5. at least two thirds of targets improve over prefix-revisit;
6. mean gauge-normalized correction/oracle cosine is at least 0.20.

Passing authorizes only one train-only causal-predictability experiment with a
single geometry target. Failure closes future-revisit distillation without
repair or validation access.

## Artifacts

- Literature boundary:
  `3D_docs/literature/future_revisit_distillation_audit.md`
- Config: `configs/EXP-071_future_revisit_teacher_premise_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp071_future_revisit_teacher_premise.py`
- Result: `revisit3d/results/EXP-071/future_revisit_teacher_premise_v10.json`
