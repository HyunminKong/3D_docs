# EXP-052 — TTT3R Metric-Alignment Premise

Status: Corrected v1.1 completed; all gates passed
Purpose: Decide whether one compact local-code space is both online-adaptable
and offline-shapeable toward absolute 3D geometry before fitting a new basis

## Question

On top of exact official TTT3R recurrence, does the existing 8-D per-token
interface (a) descend the one deployed self-supervised loss, (b) contain a
same-budget direction that improves one fixed absolute 3D objective, and (c)
permit a finite exact meta-gradient of that objective through the online step?

## Frozen train-only protocol

Use four scene-disjoint train sequences from the EXP-051 manifest:
`pumpkin/seq-01`, `heads/seq-01`, `chess/seq-01`, and `stairs/seq-01`. For each,
use four deterministic target indices `63, 127, 191, 255` and replay the four
frames ending at that target from a reset TTT3R state. This yields 16 anchors.
No validation or terminal path may be decoded.

The deployment candidate remains one generic orthonormal 8-D code basis, one
symmetric canonical-point consistency loss against the preceding frame, and
one RMS-normalized step of `0.001`. At zero code, compare current online and
metric gradients. The metric diagnostic is the median-scale-aligned relative
3D point EPE against registered 7Scenes depth in the RGB camera frame. This is
one objective, not a weighted collection of SILog/AbsRel/EPE terms.

Three endpoints are measured from the identical zero-code output:

1. the deployed online-loss step;
2. a ground-truth metric-gradient oracle step in the same code coordinates and
   with the same norm (offline feasibility only);
3. the exact gradient of realized post-online-step metric loss with respect to
   the shared basis.

The metric oracle is never a deployable method. It asks only whether the small
code space contains a useful direction. The exact meta-gradient asks whether
offline learning can rotate that space so the online direction approaches it.
No parameter update or checkpoint is created in EXP-052.

## Registered gates

- exact 16-anchor/4-scene coverage and zero-code parity within `1e-5`;
- positive online-loss gain within every scene;
- positive same-norm metric-oracle gain within every scene;
- finite, nonzero exact basis meta-gradient at every anchor.

The generic online step's absolute-metric gain and online/metric gradient cosine
are diagnostics, not gates: their mismatch is the phenomenon the proposed
offline training is intended to repair. Passing authorizes one fixed basis fit
on train data. Failure stops this interface/objective pair before validation.

## Artifacts

- Config: `configs/EXP-052_ttt3r_metric_alignment_premise_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp052_ttt3r_metric_alignment_premise.py`
- Result: `revisit3d/results/EXP-052/ttt3r_metric_alignment_premise_v10.json`

## v1.0 train-depth correction note

The v1.0 runner completed the eight registered pumpkin/heads anchors and then
stopped before chess model inference because `chess/seq-01` had raw Kinect
depth but not the derived RGB-camera registered depth required by the official
7Scenes evaluator. No result artifact was written. The first eight technical
rows showed online descent, metric-oracle improvement, and finite exact
meta-gradients, but are not treated as a completed result.

v1.1 keeps every sequence, frame, objective, step, and gate fixed. It applies
the repository's official calibration only to the 64 already selected train
context files, records raw/derived hashes, and reruns all 16 anchors. No
validation or terminal file is read or generated.

## Result

All registered checks pass on 16 anchors and four train scenes. Zero-code parity
is exact. The online loss decreases in every scene by a scene-balanced mean of
`4.92e-5`. The same-norm metric oracle improves relative 3D point EPE in every
scene by `5.43e-5` on average, and every exact basis meta-gradient is finite and
nonzero (mean norm `1.78e-4`). Peak allocated memory is 36.47 GiB.

The generic online direction exposes the intended problem: mean online/metric
gradient cosine is only `0.0253`, six of 16 anchors conflict, and six of 16
online steps harm the metric. The mean metric gain is slightly positive
(`2.21e-6`) but `pumpkin` is negative. Thus a useful direction exists inside
the compact code while the generic online direction is poorly aligned with it.
One train-only basis fit through the exact one-step metric objective is
authorized; this result does not justify validation access by itself.
