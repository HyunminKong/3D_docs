# EXP-050 — Current-Only Exact-Meta TTT Absolute Geometry Audit

Status: v1.0 completed with stale-baseline guard failure; v1.1 correction running
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

## v1.0 implementation correction note

The complete v1.0 run reproduced exact cached-readout parity internally but did
not reproduce the older EXP-036 metric artifact. A GT-free long-stream audit
then established bit-exact native/step-wise geometry on all 2,228 frames. The
method comparison is therefore internally valid, but the old TTT3R rows cannot
be mixed with the current input bytes. No v1.0 result is overwritten.

The v1.1 correction freezes a content inventory of all 4,472 current TUM RGB-D
files (digest `00c881889c9847b553adec26af459072b2247ed626111d857e7496ee3fff8ef7`),
reruns official CUT3R/TTT3R on those bytes, verifies its CUT3R rows against the
v1.0 internal base, and recomputes only the previously registered gates. The
exact-meta predictions, basis, step, policies, metrics, and hypotheses do not
change.

- Long parity diagnosis: `revisit3d/results/EXP-050/long_replay_parity_diagnosis_v10.json`
- Input inventory: `revisit3d/results/EXP-050/tum_input_inventory_v11.json`
- Native correction config: `configs/EXP-050_native_baseline_correction_v11.yaml`
- Native correction result: `revisit3d/results/EXP-050/native_cut3r_ttt3r_correction_v11.json`
