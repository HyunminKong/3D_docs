# EXP-016 — Unified Utility Address

Status: Completed; one statistical gate failed narrowly

## Question

Can one source-safe linear score both retrieve and decide reuse of the frozen EXP-015 local atom?

## Minimal deployable model

For pooled 64-D current/source descriptors `c,s`, fit one standardized Ridge regressor to future utility using `[c,s,c-s,c*s]`. Its score factorizes exactly into a current-dependent 64-D maximum-inner-product query against each stored source. Runtime retrieves top-1 and reuses it iff predicted utility is positive. There is no K sweep, fine router, risk head, PCA, or calibrated threshold.

## Causal/source-safe protocol

Replay all 557 unique train contexts in capture-time order. Before each of 218 target writes, sample a deterministic uniform panel of at most 64 prior memories and compute offline future utility with the frozen EXP-015 head. Query frames label utility only and never enter the online state.

Cross-fit by official location. For a held location, remove every training pair whose target **or source** belongs to it. The deployable policy is compared with uniform-random and appearance top-1 under the exact same accept/reject targets, plus panel oracle.

The gate requires positive utility association in every location, component-mean policy utility above 0.5%, at least 20% acceptance, no more than 20% harm, higher utility than appearance, and a positive component-bootstrap lower bound over matched-acceptance random selection. Passing freezes one all-train Ridge/MIPS artifact; failure ends the unified-address candidate.

## Result

The visual-only score reached `+0.771%` component-mean utility, 18.52% harm, and 94.72% acceptance. It beat appearance with a positive CI and had positive pair association in all four locations. Unified minus matched-acceptance random was `+0.00216`, 95% CI `[-0.00011, +0.00449]`; only this registered gate failed. No artifact was produced. EXP-017 tests one adaptation-history scalar under identical gates.

## Files

- Config: `configs/EXP-016_unified_utility_address_v10.yaml`
- Fitter/evaluator: `revisit3d/scripts/fit_exp016_unified_utility_address.py`
- Result: `revisit3d/results/EXP-016/stage0_unified_address_train_v10.json`
