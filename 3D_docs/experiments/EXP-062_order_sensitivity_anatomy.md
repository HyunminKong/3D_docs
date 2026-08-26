# EXP-062 — Fixed-Evidence Order-Sensitivity Anatomy

## Status

Completed; all registered gates passed.

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

## Result

- Exact coverage: 4 scenes, 16 contexts, 96 order evaluations, 16 replays.
- Chronological replay maximum point difference: exactly zero.
- Mean chronological/best/worst EPE: `0.079914 / 0.076251 / 0.086301`.
- Mean absolute order range: `0.010050`, stratified context-bootstrap 95% CI
  `[0.006498, 0.014046]`.
- Aggregate relative range: `12.5757%` versus registered `10%`.
- Contexts at or above 5% range: `12/16 = 75%`, exactly meeting the gate.
- Scene absolute ranges: chess `0.011547`, heads `0.005938`, pumpkin `0.005814`,
  stairs `0.016900`; all positive.
- Label-free prediction dispersion versus absolute metric range: Spearman
  `0.8353` versus registered `0.5`.
- Chronological order is best in 5/16 contexts and worst in 4/16.
- Peak allocated GPU memory: 4.61 GiB.
- Result SHA-256:
  `dcca047f004335d39dae264ab619518f022a947be21a43458c0dcdbb37c7ceb1`.

## Interpretation

Changing neither evidence nor the first anchor is enough to move fixed-query
absolute geometry materially. The strong dispersion association makes the
effect observable without RGB-D labels. This supports H20's phenomenon.

It does not make a generic swap loss novel: SIRE already introduced recurrent
permutation regularization. The next experiment must test the narrower 3D claim
that decoded geometry is the correct quotient-level commutator and that
symmetrizing state does not merely collapse it.

## Conclusion

All registered gates pass. EXP-063 is authorized as a no-fit capacity and
collision-boundary audit; model training and validation remain closed.
