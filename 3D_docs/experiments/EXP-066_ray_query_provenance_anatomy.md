# EXP-066 — RGB-Free Ray-Query Evidence-Provenance Anatomy

Status: Completed; gate failed

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

The immutable run completed all 16 contexts with exact replay and no
validation/terminal access. Only 3 of 13 gates passed.

| Quantity | Result |
|---|---:|
| Valid patch evaluations | 8,482 |
| Supported / unsupported fraction | 85.88% / 14.12% |
| Supported EPE | 0.13744 |
| Unsupported EPE | 0.14928 |
| Relative unsupported error gap | +8.62% |
| Gap bootstrap 95% CI | `[-0.00535, 0.06878]` |
| Provenance/error Spearman | 0.196 |
| Confidence/error Spearman | 0.343 |
| Spearman advantage | -0.147 |
| Advantage bootstrap 95% CI | `[-0.360, -0.185]` |
| Confidence AURC | 0.09139 |
| Equal-rank combined AURC | 0.09841 |
| Relative combined AURC gain | -7.68% |
| Combined-gain bootstrap 95% CI | `[-0.00657, -0.00171]` |
| Maximum replay difference | 0 |

Unsupported geometry is harder only weakly in aggregate and not consistently:
the error gap is negative in `pumpkin`, `heads`, and `chess`, and positive only
in `stairs`. `stairs` also has only 4.38% unsupported support, missing the 5%
coverage gate. The predicted nearest-history distance has positive association
with error in every scene but is substantially weaker than native confidence.
Its equal-rank fusion significantly worsens rather than improves selective
risk. This is not an oracle-pose or replay failure; it rejects the fixed
predicted-geometry provenance signal and the stronger binary-support premise in
this carrier/protocol.

## Conclusion

H22 is rejected as a paper center. No alternate distance, tolerance, context,
fusion weight, threshold, head, validation run, or memory system is authorized
from these exposed contexts.

Result hashes:

- result: `a04cca8fc194998d1ec58a51f0f933569480bca8b16b3afc04df5ea25fa37076`
- depth preparation: `97db5ce74f37afb591149c66cf4cc7db8dee44a283dda8b0ff1d58c5f471df33`
