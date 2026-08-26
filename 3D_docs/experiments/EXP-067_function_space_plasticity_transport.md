# EXP-067 — Function-Space Plasticity Transport Premise

Status: Completed; gate failed

Protocol: v1.0

Date: 2026-08-26

## Question

Does transporting the 3D displacement induced by a source local-TTT step and
pulling it back through the target decoder yield reusable adaptation where
direct transport of the identical 8-D code fails?

## Frozen protocol

- Data: 16 train-only physical-revisit pairs, four each from `pumpkin/seq-07`,
  `heads/seq-02`, `chess/seq-04`, and `stairs/seq-04`.
- Pair selection used pose files only: source/target gap at least 50 frames;
  rank by translation metres plus `0.01 * rotation degrees`; greedily require
  targets 40 and sources 20 frames apart. No RGB, depth, or model output was
  opened for selection.
- Causal input per pair: source predecessor, source, target predecessor,
  target. All four RGB observations update frozen TTT3R recurrence.
- Local coordinate: unchanged fixed orthonormal 8-D per-patch code and one
  normalized step (`0.001`) on symmetric predicted canonical-3D consistency.
- Source record: predicted canonical patch coordinate, source code, and
  `P_source(code)-P_source(0)` at each patch. It stores no RGB/feature/surface
  prediction beyond the coordinate needed for 3D transport.
- Direct-code control: nearest-3D transport of the source 8-D code, added after
  one current target step.
- Function-space candidate: nearest-3D transport of source output displacement;
  one target gradient step minimizes its pointwise 3D displacement residual,
  starting from the same one-current-step code.
- Controls: equal-compute second current consistency step, untransported
  function payload, and a spatial permutation of the transported payload.
- Offline metric: target self-view pointmap, median-scale-aligned relative 3D
  EPE on common valid RGB-D pixels. GT pose is pair-selection metadata only.

No parameter, basis, step, address, threshold, or router is fit. Pair identity
is an oracle capacity condition, not a deployable retrieval result.

## Frozen success gate

All must hold:

1. exactly 16 pairs/four scenes and zero-code parity at most `1e-6`;
2. source, first-current, second-current, and function pull-back objectives
   descend at every pair;
3. function transport beats equal-compute second-current TTT in every scene,
   with positive scene-bootstrap lower bound, at least 1% relative aggregate
   gain, and at most 25% pair harm;
4. function transport beats direct code transport in every scene with positive
   bootstrap lower bound;
5. function transport beats untransported and spatially shuffled function
   payloads in every scene, each with positive bootstrap lower bound;
6. no fitting or validation/terminal access.

Failure stops function-space transport without optimizer, step, loss, pair,
Jacobian, or routing repair. Success authorizes only a compact deployability
design.

## Configuration and outputs

- Config: `configs/EXP-067_function_space_plasticity_transport_v10.yaml`
- Preparation: `revisit3d/results/EXP-067/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-067/function_space_plasticity_transport_v10.json`

## Result

The immutable run completed all 16 train-only revisit pairs. Seven of 18 gates
passed and 11 failed.

| Quantity | Result |
|---|---:|
| Mean second-current EPE | 0.0684103 |
| Function gain vs second current | `2.04e-6` |
| Relative gain vs second current | 0.00299% |
| Gain 95% CI | `[-1.12e-6, 5.22e-6]` |
| Harm fraction | 43.75% |
| Gain vs direct code | `-9.36e-7` |
| Gain vs untransported function | `-1.12e-6` |
| Gain vs spatial shuffle | `1.90e-7` |
| Mean source displacement norm | `1.55e-4` |
| Maximum zero-code difference | 0 |

The point estimate beats equal current compute only negligibly, is negative in
`pumpkin`, and its interval crosses zero. Function transport loses on average
to direct code and untransported function payloads and is tied with spatial
shuffle. All source and current consistency steps descend, but the fixed
normalized pull-back step overshoots its much smaller desired displacement in
all 16 pairs, so even the function residual itself increases. This is a
registered failure, not permission for a step-size or optimizer repair.

## Conclusion

H23 is rejected for the frozen operator. Together with EXP-040--060, the result
closes compact reuse of local TTT experience on this carrier in both coordinate
and output-function forms. No line search, magnitude normalization, Jacobian
solver, basis training, routing, address, bank, or validation run is authorized
from these exposed pairs.

Result hashes:

- result: `53ab447444fb3d5c582d8c5290d6cacf8a07eb5947b1716e6a92a526355dfc65`
- depth preparation: `6b5dab2152f237e25744533b63bc2c82a0731960f865bb4fb17f99676c80e7a6`
