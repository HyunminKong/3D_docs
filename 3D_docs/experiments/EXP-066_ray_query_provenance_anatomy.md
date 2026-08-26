# EXP-066 — RGB-Free Ray-Query Evidence-Provenance Anatomy

Status: Registered before execution  
Protocol: v1.0  
Date: 2026-08-26

## Question

When CUT3R writes four causal RGB observations and receives a later ray-only
query without RGB or state update, are query surfaces unsupported by prior
visibility less accurate, and does predicted 3D history support identify that
risk beyond native confidence?

## Protocol

- Data: four roles already classified as 7Scenes train: `pumpkin/seq-03`,
  `heads/seq-01`, `chess/seq-03`, and `stairs/seq-03`.
- Contexts: query frames `[135,235,375,475]`; history offsets
  `[-40,-30,-20,-10]`; query offset `0` never supplies RGB.
- Carrier: frozen CUT3R checkpoint, native `cut3r` update mode, 512x384. Four
  RGB views update state. A GT-camera raymap queries the fifth view with
  `img_mask=false`, `ray_mask=true`, and `update=false`.
- Evaluation: 768 patch centers. A GT query point is historically supported
  when projection is in-frame and registered history depth agrees within
  `max(0.05 m, 5% depth)`. This is an offline label only.
- Metric: median-scale-aligned relative 3D EPE in the first-camera frame.
- Predicted provenance risk: nearest 3D distance from each predicted query
  patch to the union of history patch points, divided by predicted query range.
- Native risk: negative confidence. Combined risk is the fixed equal mean of
  within-context ranks. No parameter, threshold, or statistic is fit.
- Statistics: scene-stratified context bootstrap, 10,000 draws.

Known pose/intrinsics construct the controlled ray query and offline visibility
label. This cannot support a deployable pose-estimation claim.

## Frozen success gate

All must hold:

1. exactly 16 contexts/four scenes and replay difference at most `1e-5`;
2. every scene contains at least 5% supported and 5% unsupported patches;
3. unsupported-minus-supported EPE is positive in every scene, its bootstrap
   lower bound is positive, and aggregate relative gap is at least 20%;
4. provenance/error Spearman is positive in every scene and at least 0.30 in
   aggregate;
5. provenance Spearman exceeds confidence Spearman by at least 0.10 and the
   paired bootstrap lower bound is positive;
6. equal-rank fusion lowers AURC versus confidence in every scene, with a
   positive paired bootstrap lower bound and at least 5% relative gain.

Failure stops this fixed signal without another distance, threshold, feature,
head, or validation access. Success authorizes only broader baseline/capacity
evaluation.

## Configuration and outputs

- Config: `configs/EXP-066_ray_query_provenance_anatomy_v10.yaml`
- Depth preparation: `revisit3d/results/EXP-066/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-066/ray_query_provenance_anatomy_v10.json`

## Result

Not run.
