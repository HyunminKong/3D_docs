# EXP-050 — Current-Only Exact-Meta TTT Absolute Geometry Audit

Status: Registered
Purpose: Decide whether the surviving compact TTT coordinate is a competitive
paper candidate without continual memory

## Question

Does one local exact-meta TTT step improve absolute 3D geometry on a competitive
frozen CUT3R carrier, beyond both the official zero-code output and an equal
8-D generic orthonormal coordinate, and does it reach the official TTT3R result?

## Frozen protocol

Use the existing EXP-043 6,144-parameter basis without fitting on the exposed
TUM data. Replay the exact EXP-036 causal stream: 2,228 frames, 111 query
targets, three sequences, official 512 preprocessing, query readout without
recurrent-state update, and sequence-only reset. The current query RGB may
create its local code from the fixed symmetric point-consistency loss, but the
code cannot update recurrent state or persist to another frame. RGB-D ground
truth is decoded only after predictions and is never an adaptation input.

Compare:

1. official CUT3R zero-code output;
2. one step in the fixed generic 8-D orthonormal coordinate;
3. one step in the frozen exact-meta coordinate (primary);
4. two exact-meta current steps (diagnostic only);
5. the already frozen official TTT3R predictions from EXP-036.

The generic and exact variants use identical code dimension, online loss, step
size, head location, carrier state, and compute graph. Only the 6,144 projection
weights differ. The two-step result cannot replace the registered one-step
primary after inspection.

## Registered gates

The replayed zero-code CUT3R metrics must reproduce EXP-036 within `1e-5`.
For SILog, aligned AbsRel, and 3D EPE, exact-meta one-step must have a positive
sequence-bootstrap lower bound over CUT3R and over the generic coordinate.
These are the method-feasibility gates.

For the stronger top-tier competitiveness gate, exact-meta one-step must also
beat official TTT3R with a positive lower bound on all three primary metrics.
Failure of the competitiveness gate means the current frozen realization is
not a standalone competitive reconstruction paper, even if its coordinate
ablation succeeds.

TUM was exposed during the earlier carrier audit, so this experiment is
development evidence only. It cannot serve as final paper generalization and
the EXP-039 terminal split remains unopened.

## Artifacts

- Config: `configs/EXP-050_current_only_exact_meta_tum_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp050_current_only_exact_meta_tum.py`
- Result: `revisit3d/results/EXP-050/current_only_exact_meta_tum_v10.json`
