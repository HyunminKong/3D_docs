# EXP-069 — Query-Integrity Ranking Premise

Status: Corrected v1.1 completed; gate failed

Protocol: v1.1 (coverage-accounting correction; v1.0 aborted before result)

Date: 2026-08-27

## Question

Does pointwise APD frequently prefer a D4RT prediction produced under a larger
irrelevant clip-context change even though that change causes a consistently
larger held-out non-gauge query-equivalence residual?

## Frozen discovery and confirmation boundary

- Discovery only: the exposed 16-sequence EXP-068 premise result. It suggested
  the ranking hypothesis after H24 had already failed its different registered
  gate.
- Confirmation: the 11 files already name-hash assigned to the `validation`
  role in `adt_cross_clip_exp068_v10.json`. Their NPZ content and model outputs
  have never been opened.
- The 11 files are reassigned once as the H25/EXP-069 premise set and can no
  longer validate H24 or select an EXP-069 method.
- The 12 terminal files remain unopened.
- Carrier, clips, source/target frames, track selection, alignment/evaluation
  halves, four depth layers, and metrics are unchanged from EXP-068.

## Frozen measurements

For each of 33 sequence/target rows:

1. compute the large-shift and adjacent-shift held-out four-layer Sim(3)
   residuals exactly as in EXP-068;
2. compute signed APD gain `APD_large - APD_reference`, where positive means
   the conventional metric prefers the large-shift context;
3. count a ranking inversion when the large-shift structural residual exceeds
   the adjacent residual while signed APD gain is positive;
4. measure Spearman association between structural damage
   `layer_large - layer_adjacent` and signed APD gain.

No row, threshold, shift, layer, target, or correlation statistic is selected
after inference.

## Frozen complete gate

All conditions must hold:

1. exact replay maximum absolute difference `<=1e-6`;
2. all 11 fixed files are attempted, at least 9 are evaluable, every evaluable
   sequence supplies all three targets, and no layer fit falls back;
3. large-minus-adjacent structural damage has positive sequence-bootstrap 95%
   lower bound and is positive in at least 80% of evaluable sequences;
4. the large-shift layer residual has positive lower bound and is positive in
   every sequence;
5. mean signed APD gain is non-negative and its sequence-bootstrap 95% lower
   bound is positive;
6. APD prefers the large-shift context in at least half of evaluable sequence
   means and at least 40% of evaluable target rows;
7. target-row Spearman magnitude between structural damage and signed APD gain
   is at most `0.30`;
8. no model fitting, threshold selection, or terminal access occurs.

Failure closes the metric-ranking hypothesis without changing the sign test,
correlation limit, clips, targets, or aggregation. Success authorizes only a
multi-model/dataset coverage decision, not a training loss.

## Artifacts

- Config: `configs/EXP-069_query_integrity_ranking_v10.yaml`
- Source role manifest: `revisit3d/manifests/adt_cross_clip_exp068_v10.json`
- Result: `revisit3d/results/EXP-069/query_integrity_ranking_v10.json`
- Corrected config: `configs/EXP-069_query_integrity_ranking_v11.yaml`
- Corrected result: `revisit3d/results/EXP-069/query_integrity_ranking_v11.json`
- Aborted-run record: `revisit3d/results/EXP-069/aborted_v10.json`
- Literature boundary:
  [query-integrity metric audit](../literature/query_integrity_metric_audit.md)

## v1.1 coverage correction before any aggregate result

The v1.0 run processed eight sequences, then found that the ninth attempted
sequence had zero annotated tracks visible at global source frame 16. It raised
before computing aggregate metrics or writing a result. The clip/source/target,
alignment minimum, metric, signed APD thresholds, correlation threshold, model,
and role are unchanged.

Version 1.1 attempts every one of the 11 fixed files and records an exclusion
when fewer than 64 eligible tracks exist for deterministic 50/50 alignment and
evaluation halves. No replacement is sampled. The corrected coverage gate
requires at least 9/11 evaluable sequences. Count gates are expressed as the
same intended proportions: structural large-over-adjacent in at least 80% of
evaluable sequences, positive layer residual in all, APD preference in at least
half of sequence means and 40% of target rows. The positive signed-APD bootstrap
and `|Spearman|<=0.30` gates are unchanged.

## Result

Version 1.1 attempted all 11 fixed files. Ten were evaluable for all three
targets; one had zero tracks visible at the immutable source frame and is
retained as a coverage exclusion. Exact replay was bitwise, no layer fit fell
back, no model was fit, and terminal remained unopened.

| Quantity | Result |
| --- | ---: |
| Evaluable sequences / targets | 10 / 30 |
| Layer-aligned large residual | 2.6613% of scene scale |
| Layer residual 95% CI | [1.5473%, 3.9907%] |
| Large minus adjacent structural damage | 1.6891% |
| Structural-damage 95% CI | [1.0044%, 2.5077%] |
| Sequences with large > adjacent | 10/10 |
| Mean signed APD gain (large - reference) | -0.01593 |
| Signed APD gain 95% CI | [-0.08154, 0.04212] |
| Sequences where APD prefers large shift | 6/10 |
| Targets where APD prefers large shift | 16/30 |
| Structural/APD target Spearman | 0.1228 |
| Complete gate | 15/17; failed |

The structural diagnostic generalizes strongly, and APD makes frequent local
ranking inversions. However, the paper premise deliberately required more than
frequency: mean signed APD gain had to be non-negative with a positive
sequence-bootstrap lower bound. Its observed mean is negative and its interval
crosses zero. These are the only two failed gates.

## Conclusion

H25 is rejected as the registered evaluation-paper premise. APD does not encode
query integrity and can prefer a structurally less stable context on individual
rows, but it does not systematically reward that context on untouched data.
This supports reporting query integrity as a complementary diagnostic, not the
strong claim that standard pointwise evaluation ranks the failure backward.

No correlation threshold, APD sign test, aggregation, coverage rule, clip, or
query is changed. No multi-model benchmark, equivalence loss, validation fit,
or terminal evaluation is authorized from EXP-069.

## Immutable hashes

- corrected result:
  `365009557e8d30d3b670299d96f4fb79e8df50df6ab5e3966eb3883025f1a4a8`
- corrected config:
  `8277cb57af56a7322f7f32df7de93a7d8619c3d3e6e67cef540303287a7bb585`
- aborted-run record:
  `fdd4b6745f80df42c951e9f4e01551ce96d7b1eb0b54deebd0364b6cd06a0095`
