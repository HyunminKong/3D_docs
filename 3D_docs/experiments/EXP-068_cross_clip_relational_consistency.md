# EXP-068 — Cross-Clip Relational Consistency Premise

Status: Completed; registered gate failed

Protocol: v1.0

Date: 2026-08-27

## Question

Does the same physical D4RT query decoded from overlapping fixed-length clips
exhibit a material non-rigid 4D inconsistency that survives global Sim(3) and
depth-layer alignment, rather than only a removable window gauge change?

## Frozen data boundary

- Carrier: released OpenD4RT 32-frame 9-dataset checkpoint, used as an
  unofficial D4RT reproduction rather than the proprietary D4RT model.
- Dataset: 39 ADT-mini sequences not used by the earlier external weakness
  study. Sequence roles are assigned deterministically from names before NPZ,
  RGB, track, or model access: 16 premise, 11 validation, 12 terminal.
- EXP-068 opens only the 16 premise sequences. Validation and terminal roles
  remain unopened.
- The ten sequences in the external `Open-d4rt/probe/` study are explicitly
  exposed and excluded.

## Frozen query protocol

- Clip A: global frames `[0, 32)`.
- Large-shift clip B: `[16, 48)`.
- Adjacent-context control C: `[1, 33)`.
- Identical physical source frame: global frame 16.
- Identical target frames: global 20, 24, and 28.
- Tracks must have finite source pixels and finite target 3D. At most 512 tracks
  are selected deterministically per sequence.
- The source RGB patch, target time, and target camera are physically identical
  across A/B/C; only encoded clip context and local timestep indices differ.
- Queries are divided deterministically into alignment and evaluation halves.
  Alignment is never fit on evaluation queries.

## Controls and metrics

1. Exact replay of clip A and identical queries.
2. Raw A--B and A--C disagreement normalized by target scene scale.
3. Held-out global Sim(3), fit on alignment queries and scored on evaluation
   queries.
4. A strong four-depth-layer oracle: separate Sim(3) transforms are fit on
   alignment queries within reference-depth quartiles and scored on held-out
   queries. This intentionally upper-bounds simple layer alignment.
5. Held-out local pair-distance disagreement after alignment, which is
   invariant to a shared rigid coordinate change.
6. Each window's independently median-scale-aligned 3D EPE and APD, to test
   whether standard pointwise metrics expose the same failure.

No model parameter, transform family, number of layers, clip offset, query
frame, threshold, or metric is selected after inference.

## Frozen success gate

All must hold on the 16 premise sequences:

1. exact replay maximum absolute difference at most `1e-6`;
2. all 16 sequences and all three target frames complete with finite metrics;
3. large-shift residual after four-layer alignment is at least `0.5%` of scene
   scale, has a positive sequence-bootstrap 95% lower bound, and is positive in
   every sequence;
4. large-shift layer-aligned residual exceeds the adjacent-shift residual with
   a positive bootstrap lower bound and is larger in at least 12/16 sequences;
5. layer alignment leaves at least 25% of the raw large-shift disagreement;
6. large-shift pair-distance disagreement has a positive bootstrap lower bound
   and is at least twice the exact-replay control;
7. the mean absolute APD difference between A and B is below `0.05`, confirming
   that the relational effect is not equivalent to a large ordinary accuracy
   collapse;
8. no validation/terminal sequence or model fitting is used.

Failure closes this candidate without alignment, offset, layer-count, query, or
threshold repair. Success authorizes a matched original-loss fine-tune versus
one cross-clip equivalence loss; it does not authorize a new architecture.

## Configuration and planned outputs

- Config: `configs/EXP-068_cross_clip_relational_consistency_v10.yaml`
- Role manifest: `revisit3d/manifests/adt_cross_clip_exp068_v10.json`
- Result: `revisit3d/results/EXP-068/cross_clip_relational_consistency_v10.json`
- Gate-accounting correction:
  `revisit3d/results/EXP-068/gate_accounting_correction_v11.json`

## Literature boundary

See
[cross-clip 4D consistency audit](../literature/cross_clip_4d_consistency_audit.md).
Generic overlapping-window alignment is occupied by Geo4D, LASER, and V-DPM;
generic rigidity regularization is also occupied. The premise is viable only if
a held-out non-rigid query-equivalence residual remains after those nuisance
families are controlled.

## Result

The frozen v1.0 inference completed all 16 premise sequences and 48 target
frames. Exact replay was bitwise (`0.0` maximum absolute difference), no layer
fit fell back, and neither validation nor terminal data was opened.

| Quantity | Result |
| --- | ---: |
| Raw large-shift disagreement / scene scale | 12.5606% |
| After held-out global Sim(3) | 4.0590% |
| After held-out four-layer Sim(3) | 2.8292% |
| Four-layer residual 95% CI | [1.7779%, 4.3952%] |
| Adjacent one-frame residual | 0.9293% |
| Large minus adjacent | 1.8999% |
| Large-minus-adjacent 95% CI | [1.0893%, 3.0913%] |
| Sequences with large > adjacent | 16/16 |
| Four-layer residual retained from raw | 34.5074% |
| Pair-distance residual / scene scale | 1.0043% |
| Pair-distance residual 95% CI | [0.5467%, 1.7284%] |
| Mean absolute A/B APD difference | 0.06051 |

The structural part of H24 is strongly supported: a large clip-context shift
changes the same physical query more than an adjacent shift, and a material
residual survives both held-out global alignment and a deliberately generous
depth-layer oracle. Every sequence has the same direction of effect.

The complete registered premise nevertheless fails. The APD-blindness gate
required mean absolute APD difference `< 0.05`, but the immutable result is
`0.06051`. Therefore the failure is not sufficiently hidden from an ordinary
pointwise metric to support the intended paper distinction. The method phase,
validation role, and terminal role are not opened.

## Accounting correction

The original v1.0 JSON reports 14/17 gates because its boolean dictionary used
the names `validation_accessed` and `terminal_accessed` with value `false`, then
counted all boolean values as pass indicators. Those two values actually prove
the source-safe gate passed. The immutable v1.0 result is retained; correction
v1.1 renames the indicators to `validation_not_accessed=true` and
`terminal_not_accessed=true`. The corrected accounting is **16/17**, with
`pointwise_apd_blindness` as the sole scientific failure. No model was rerun,
and no metric, threshold, or measured value changed.

## Conclusion

H24 is rejected as the registered top-tier paper premise, while its narrower
diagnostic observation is retained: independently encoded overlapping clips
produce a real, non-gauge query-equivalence residual. Because generic temporal
window alignment and relational regularization are already occupied, and the
registered claim that APD misses the problem failed, this evidence does not
authorize an equivalence loss or a paper method. Offset, layer count, APD
threshold, query choice, and alignment may not be repaired on these exposed
premise sequences.

## Immutable artifact hashes

- v1.0 result:
  `112c74683b2898641caed72ea551e61ad8a613f799b7b532180d9345f7255328`
- config:
  `a4f5ea3adcaae3e43f844230666629864bb222d06ea04782607f0643acfcfb1f`
- role manifest:
  `42293e5fece6a199d14317a8b75c90e766835110155d10395efd594c549d0b0d`
- released checkpoint:
  `1f63305422fdc2000b057fbbc1d37459ac1a8063bbfcd0e3b7d473f5485943f5`
