# EXP-029 — Metric-Utility Address

Status: Registered; not yet executed

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
