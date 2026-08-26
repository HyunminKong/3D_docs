# EXP-056 — Pairwise Geometry Residual Observability

Status: Completed; registered gate failed and no checkpoint was created
Purpose: Test whether current/previous predicted-geometry residuals expose the
token-axis oracle structure that current tokens failed to learn

## Data boundary

Reuse exactly the 16 already exposed EXP-052/054 anchors: four anchors from
each of `pumpkin`, `heads`, `chess`, and `stairs`. Every prediction is a
four-frame reset sequence in official TTT3R mode. Use leave-one-scene-out folds;
RGB-D-derived oracle labels from the held scene may never fit its predictor.
Validation and both terminal partitions remain closed.

## Features and label

The offline binary label for token `n`, axis `j` is
`g_online[n,j] * g_metric[n,j] > 0`. It is used only to fit and score this
diagnostic. The source-safe feature sets are:

1. `token`: layer-normalized frozen current decoder patch token;
2. `geometry`: normalized 3D residual to the nearest previous predicted
   canonical point plus log relative residual magnitude;
3. `combined`: concatenation of token and geometry;
4. `shuffled_geometry`: combined input after a deterministic within-anchor
   spatial permutation of the geometry rows.

For each OOF fold and feature set, standardize columns on the three training
scenes and compute one moment-linear weight matrix
`mean(x_standardized * label_sign)`. Normalize each output score by its train
RMS and convert it to an axis scale with `1 + tanh(score)`. There is no
optimizer, regularization coefficient, calibrated threshold, or saved model.

## Functional evaluation

Apply the OOF axis scales to both the online code gradient and code readout,
using the unchanged normalized `0.001` step. Compare global, token, geometry,
combined, shuffled-geometry, and the offline token-axis oracle. Report balanced
label accuracy as a diagnostic, but decide using realized median-scale-aligned
relative 3D point gain.

## Registered gates

- exact 16-anchor/four-fold coverage, finite values, zero-code parity at most
  `1e-5`, and no validation/terminal access;
- combined balanced label accuracy exceeds token-only and shuffled geometry;
- combined online loss descends and metric gain is positive in every scene;
- combined metric gain beats global, token-only, and shuffled geometry in
  every scene and with positive paired anchor-bootstrap 95% intervals;
- combined harm is at most 25%.

Passing supports H16 and authorizes registration of one final pairwise
conditioner fit. It does not authorize validation or produce a method
checkpoint. Failure stops this minimal residual feature realization without
feature, score, or optimizer tuning.

## Artifacts

- Config: `configs/EXP-056_pairwise_geometry_observability_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp056_pairwise_geometry_observability.py`
- Result: `revisit3d/results/EXP-056/pairwise_geometry_observability_v10.json`

## Result

All 16 anchors and four leave-one-scene-out folds completed with exact
zero-code parity, finite values, positive combined online descent in every
scene, and no validation or terminal access.

The combined token/residual score did not predict the oracle labels. Its mean
balanced accuracy was `50.204%`, versus `50.206%` for token-only, `49.483%` for
geometry-only, and `50.240%` after spatially shuffling geometry. The residual
therefore supplies no scene-general label information under this linear
diagnostic.

Realized combined metric gain was `2.244e-6`, compared with `0.809e-6` for
global, `0.264e-6` for token-only, and `0.965e-6` for shuffled geometry. The
combined-minus-token interval was positive, but combined-minus-global CI
`[-0.853e-6, 3.846e-6]` and combined-minus-shuffle CI
`[-1.858e-6, 4.887e-6]` crossed zero. `pumpkin` remained negative and combined
harm was 50%. By contrast, the offline oracle retained `20.386e-6` mean gain.

The registered gate fails. H16 is rejected for this minimal residual, no
method checkpoint exists, and D141 stops feature or classifier repair on these
anchors.
