# EXP-007 — Continual Atom Bank and Consolidation

Status: **Starting; Stage-0 protocol design.**

## Question

Can a causally written, capacity-bounded bank of visually addressable plasticity atoms retain the future-utility benefit of an unbounded bank without increasing negative transfer?

## Hypothesis

H5: adaptation-regime-aware consolidation can preserve most routed revisit utility with sublinear memory growth. Place identity alone is not the consolidation target.

## Leakage boundary

- Stage 0 and all architecture selection use the expanded train split only.
- An atom is retrievable only after its context has appeared in the simulated stream.
- Query/future frames measure utility only and never enter writes, keys, router features, merge, eviction, or selection.
- EXP-006 validation is closed to EXP-007 tuning; the previously exposed test split remains prohibited.
- Stream order, capacities, baselines, and success criteria must be fixed before any future EXP-007 holdout is opened.

## Stage 0 — causal-bank feasibility before learned consolidation

1. Construct deterministic component-safe train streams from observed contexts.
2. Write source/current adaptation atoms only after they become causally available.
3. Compare full-bank oracle, appearance-top-K oracle, and the frozen EXP-006 router.
4. Measure bank growth, retrieval compute, utility, regret, deadband harm, and coverage.
5. Sweep capacity only on train for FIFO, reservoir, appearance-diversity coreset, and utility-history eviction.
6. Quantify the capacity needed to retain 90% and 95% of unbounded routed utility.

Stage 0 does not train a merge network. It first asks whether useful redundancy exists and whether a bounded bank is scientifically plausible. Merge/eviction learning is admitted only if the causal full-bank/top-K signal remains positive and a non-trivial capacity reduction exists.

## Required controls

- no memory / one current TTT step;
- unbounded all-write bank;
- FIFO;
- uniform reservoir;
- appearance-only diversity;
- place/scene deduplication;
- oracle capacity subset upper bound;
- random retrieval and visual mean where computationally comparable.

## Metrics

- normalized future utility and regret;
- directional and component deadband harm;
- raw-sign harm;
- accept/reject rate;
- top-K recall of the all-bank utility oracle;
- retained utility relative to unbounded router and oracle;
- records, bytes, candidate comparisons, wall time, and peak GPU memory;
- write, merge, eviction, and reactivation counts.

## Current next action

Implement a train-only all-memory utility-table probe with causal availability masks and deterministic stream orders. Use it to choose whether EXP-007 should focus on merge, eviction, or candidate prefiltering; do not build a learned consolidation network first.
