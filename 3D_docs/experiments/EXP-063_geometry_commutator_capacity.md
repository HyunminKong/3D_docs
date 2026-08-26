# EXP-063 — Geometry-Decoded Commutator Capacity Audit

## Status

Completed; complete gate failed.

## Question

Is decoded 3D disagreement a more geometry-relevant measure of recurrent path
noncommutativity than normalized latent-state distance, and does the symmetric
barycenter of the six recurrent paths remain geometry-healthy?

## Collision boundary

SIRE already minimizes pairwise swapped-path distance in generic recurrent
models. EXP-063 cannot establish novelty for a latent commutator. It tests the
narrower claim that a fixed-query pointmap quotient is the relevant object in
streaming 3D, because latent equality is neither necessary nor sufficient for
surface equality.

## Protocol

- Reuse exactly the frozen EXP-062 16 train contexts and six paths; no new data,
  fitting, validation, or terminal access.
- Reproduce all six query pointmaps and metric errors.
- Retain each final recurrent `state_feat` and pose-retrieval `mem` before the
  non-updating query.
- Primary latent dispersion: mean pairwise RMS `state_feat` distance divided by
  the chronological state's RMS magnitude. Pose-memory dispersion is
  descriptive only.
- Geometry dispersion: the already registered mean pairwise query-point
  distance after each pointmap is divided by its median predicted query depth.
- Form one permutation-barycenter state by arithmetic-averaging all six
  `state_feat` and `mem` tensors; immutable positions/initial state come from
  the chronological path. Read the identical query with `update=false`.
- Score every individual path and barycenter on one common valid query mask,
  with the same independent median-depth scale alignment as EXP-062.

## Frozen success gate

All must hold:

1. EXP-062 per-order EPE and geometry dispersion reproduce within `1e-5`;
2. geometry-dispersion Spearman with absolute metric range is at least `0.5`;
3. geometry association exceeds primary latent-state association by at least
   `0.20`;
4. barycenter-state EPE improves over mean six-path EPE in every scene with a
   positive stratified context-bootstrap interval;
5. barycenter-state aggregate EPE does not exceed chronological aggregate EPE.

The output-pointmap barycenter is reported only as a descriptive ensemble upper
control. Passing authorizes a fresh train-only geometry-commutator
trainability design. Failure prevents a generic SIRE-style adaptation from
being presented as the paper method.

## Artifacts

- Config: `configs/EXP-063_geometry_commutator_capacity_v10.yaml`
- Result: `revisit3d/results/EXP-063/geometry_commutator_capacity_v10.json`

## Result

- EXP-062 EPE and geometry dispersion reproduce exactly.
- Geometry dispersion versus absolute range: Spearman `0.835294`.
- Normalized latent-state dispersion versus range: `-0.011765`.
- Geometry-over-latent association: `0.847059`, passing its gate.
- Pose-memory dispersion versus range: `0.497059` (descriptive).
- Six-order mean / chronological / state-barycenter EPE:
  `0.081118 / 0.079914 / 0.082279`.
- State-barycenter gain over order mean: `-0.001162`, 95% CI
  `[-0.002499, -0.000053]`; negative in chess, heads, and stairs.
- Output-pointmap barycenter EPE: `0.080265`; it improves the six-order mean by
  `0.000852` in all 16 contexts and all scenes, but remains `0.000351` worse
  than aggregate chronological EPE.
- Peak allocated GPU memory: 4.65 GiB.
- Result SHA-256:
  `70264df08acda90a5f6731c9136e196c5e9e9439dad75d58fbef1205b4db854a`.

## Interpretation

Internal state distance is not the right measure of geometry-relevant
noncommutativity. Forcing latent paths toward their arithmetic center risks
state collapse and is not justified. The positive output-barycenter control
localizes the remaining possibility to a decoded geometry quotient, but does
not yet show benefit for the deployable chronological path.

## Conclusion

The registered experiment fails. Generic latent commutator regularization is
stopped. D152 authorizes only EXP-064's fixed local output-direction audit.
