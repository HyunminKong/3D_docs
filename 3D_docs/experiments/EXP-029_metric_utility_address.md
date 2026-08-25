# EXP-029 — Metric-Utility Address

Status: Completed; all registered gates passed

## Question

Can one factorized linear address predict which causal past EXP-028 atom will
improve future aligned-log geometry, without using query frames or LiDAR online?

## Protocol

Freeze the EXP-028 atom. Traverse the existing train contexts in timestamp
order, write only after prediction, and sample a deterministic panel of at most
64 causal past records. Sparse query LiDAR provides the single offline utility
label

\[
U=(L^{cur}_{log}-L^{reuse}_{log})/(|L^{cur}_{log}|+\epsilon).
\]

Fit one Ridge score on `[current, source, current-source, current*source]` with
leave-one-location-out target folds and removal of every held-location source.
The score factorizes exactly into MIPS. Top-1 is reused only when predicted
utility is above semantic zero. There is no risk head, learned threshold, fine
router, or new inference loss.

## Registered gate

Every held location must have positive pairwise Spearman association. Selected
utility must be positive, acceptance at least 20%, harm at most 30%, exceed
appearance at matched acceptance, and beat same-panel random with a positive
physical-component bootstrap interval. Failure creates no artifact and blocks
terminal evaluation.

## Files

- Config: `configs/EXP-029_metric_utility_address_v10.yaml`
- Fitter: `revisit3d/scripts/fit_exp029_metric_utility_address.py`
- Result: `revisit3d/results/EXP-029/stage0_metric_utility_address_train_v10.json`
- Conditional artifact: `revisit3d/checkpoints/exp029_metric_utility_address_v10.joblib`

## Result

The causal table contains 13,631 target/source pairs, 218 unique targets, and
25 physical components. Pair OOF Spearman was 0.2597. Every held location was
positive: Boston 0.1557, Holland Village 0.2749, One North 0.4037, and
Queenstown 0.2148.

| Policy | metric utility | harm | acceptance |
|---|---:|---:|---:|
| unified metric address | +0.00320 | 11.44% | 93.09% |
| matched random | +0.00122 | 16.17% | 93.09% |
| appearance | +0.00125 | 26.87% | 93.09% |
| panel oracle | +0.00515 | 0% | 99.94% |

Unified-minus-random component CI was `[0.00093, 0.00326]`; unified-minus-
appearance was `[0.00082, 0.00335]`. The artifact hash is
`d8b81fff36d5cb5635c194a63b422edf700c0683b7f7eb2d477be67091430984`.

## Conclusion

One source-safe factorized linear score predicts future metric utility and
substantially reduces raw-candidate harm without a risk head or learned
threshold. This passes address feasibility. A no-fit full-system OOF geometry
audit is still required before the locked EXP-021 terminal benchmark is opened.
