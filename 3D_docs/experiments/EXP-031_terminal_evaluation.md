# EXP-031 — Immutable Official-Test Terminal Evaluation

Status: Completed; registered gate failed on coverage only

## Frozen candidate

- EXP-028 atom checkpoint SHA-256
  `3ebf194f3a28876014e46d1d3bbdbcd1422cfb8ebdba48f3d16635520ca787ae`;
- EXP-029 address SHA-256
  `d8b81fff36d5cb5635c194a63b422edf700c0683b7f7eb2d477be67091430984`;
- one `track3D` step, eta 0.0125, 8-D code, visual transport, residual 0.10;
- top-1 metric-utility address with semantic-zero acceptance;
- write after prediction and deterministic reservoir capacity 64 per official
  location.

## Terminal protocol

The locked EXP-021 manifest contains 214 directional episodes, 96 scenes, 29
physical components, and three locations. Stage 0 converts only selected scene
metadata. Stage 1 decodes RGB and creates frozen foundation/tracker outputs
using train-fitted PCA. Stage 2 evaluates causal full memory, current-only,
same-bank random expectation, and appearance retrieval; sparse LiDAR is read
only after the online decision for geometry metrics.

No terminal observation, prediction, or metric may change the model, address,
threshold, capacity, control, or success rule. This experiment is terminal
evidence, not model selection.

## Registered gate

At least 190 unique targets and 25 components must be evaluable. The full
policy must improve mean SILog, aligned AbsRel, and 3D EPE over current-only,
matched random, and appearance, with at least one positive component interval
in each comparison family. Any failure is reported without repair.

## Files

- Config: `configs/EXP-031_terminal_evaluation_v10.yaml`
- Geometry cache: `revisit3d/cache/EXP-031/terminal_geometry_v10.pt`
- Result: `revisit3d/results/EXP-031/stage2_terminal_evaluation_v10.json`

## Result

The immutable evaluator ran once. It produced 187 unique targets across all 29
components. Full memory improved every registered primary mean over every
control:

| Comparison | SILog improvement | aligned AbsRel improvement | 3D EPE improvement |
|---|---:|---:|---:|
| full vs current | +0.05492 | +0.001752 | +0.01446 m |
| full vs random | +0.02347 | +0.000389 | +0.00604 m |
| full vs appearance | +0.02634 | +0.000332 | +0.00707 m |

All three component-bootstrap intervals were positive against current and
random. SILog and 3D EPE intervals were positive against appearance; aligned
AbsRel had positive mean but CI `[-0.000118, 0.000738]`. This satisfies the
registered method-comparison rules.

The overall gate nevertheless failed because 187 is below the registered
minimum of 190 targets. EXP-032 establishes that 187 was the maximum possible
under the frozen evaluator: 214 directional episodes collapse to 188 unique
target contexts, and the first target in one location has an empty causal bank.
Every causally evaluable target and all 29 components were included. The
registered failure remains a failure and is not repaired post hoc.

## Conclusion

The terminal data support the method-effect claim but not a literal statement
that EXP-031 passed all preregistered gates. Paper language must report both:
all frozen geometry comparisons favored the full policy, while the coverage
gate was invalidly specified in episode units rather than unique causal-target
units.
