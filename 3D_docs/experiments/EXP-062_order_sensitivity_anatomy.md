# EXP-062 — Fixed-Evidence Order-Sensitivity Anatomy

## Status

Preregistered; not yet run.

## Question

Does the frozen TTT3R recurrent update produce materially different absolute
query geometry when exactly the same static observations are assimilated in a
different order?

## Data contract

- Frozen EXP-051 7Scenes train role only; validation and terminal unopened.
- Sequences: `pumpkin/seq-03`, `heads/seq-01`, `chess/seq-03`,
  `stairs/seq-03`.
- Query targets: 95, 175, 335, 415.
- Each context contains four history frames and one fixed query: 16 contexts.
- The first history frame remains first. The next three history frames use all
  six permutations. Query RGB/depth/pose and the history multiset are identical.
- Query has `update=false`; RGB is online input, while RGB-D labels only score
  offline absolute geometry.

## Protocol

1. For every order, reset from the same empty state and process the fixed first
   history plus one permutation of the same middle triplet.
2. Read the same fifth query without updating persistent state.
3. Score `pts3d_in_self_view` against query camera-frame RGB-D points after an
   independent median depth scale alignment. This removes arbitrary scale but
   cannot remove spatial geometry differences.
4. Repeat the chronological order once and record maximum output difference.
5. For each context, compute best, worst, chronological EPE and relative range
   `(worst-best)/chronological`.
6. Compute an RGB-D-free dispersion score: the mean pairwise relative point
   distance among the six query predictions after normalizing each by its own
   median predicted depth.
7. Test association between dispersion and metric order range across contexts.

## Frozen success gate

All must hold:

1. exactly 4 scenes, 16 contexts, 96 order evaluations, and 16 replay checks;
2. chronological replay maximum absolute point difference `<=1e-5`;
3. mean worst-best EPE is positive in every scene and stratified context-
   bootstrap 95% lower bound is positive;
4. aggregate relative order range is at least 10%;
5. at least 75% of contexts have relative range at least 5%;
6. Spearman correlation of label-free dispersion with metric range is at least
   0.5.

Passing authorizes one train-only, no-fit commutator-capacity experiment.
Failure closes H20 without selecting favorable orders or tuning these gates.

## Artifacts

- Config: `configs/EXP-062_order_sensitivity_anatomy_v10.yaml`
- Depth preparation:
  `revisit3d/results/EXP-062/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-062/order_sensitivity_anatomy_v10.json`
