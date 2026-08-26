# EXP-069 — Query-Integrity Ranking Premise

Status: Registered; not yet run

Protocol: v1.0

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
2. exactly 11 sequences and 33 targets complete, with no layer fallback;
3. large-minus-adjacent structural damage has positive sequence-bootstrap 95%
   lower bound and is positive in at least 9/11 sequences;
4. the large-shift layer residual has positive lower bound and is positive in
   every sequence;
5. mean signed APD gain is non-negative and its sequence-bootstrap 95% lower
   bound is positive;
6. APD prefers the large-shift context in at least 6/11 sequence means and at
   least 14/33 target rows;
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
- Literature boundary:
  [query-integrity metric audit](../literature/query_integrity_metric_audit.md)
