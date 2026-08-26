# AGENTS.md

## Project

Compact Streaming 3D Reconstruction Research.

The former continual adaptation-memory, gauge/local-reliability,
order-robustness, and calibration-shock lines are preserved through EXP-065
and are inactive. H22 and H23 were rejected by EXP-066/067. Coordinate- and
function-space local TTT reuse are both closed on the competitive carrier. Only
explicit retained surface content has strong causal evidence, but its generic
memory formulation is occupied and no natural trigger exists. No accepted
method, bank, router, or learned component exists.

This workspace contains the active research implementation plus several external reference repositories. The repository documents, not chat history, are the source of truth.

## Before starting any task

Read, in order:

1. `3D_docs/research_state.md`
2. `3D_docs/hypothesis.md`
3. `3D_docs/decisions.md`
4. `3D_docs/experiments/index.md`

For method or architecture work also read `3D_docs/method.md`. For literature work start from `3D_docs/literature/index.md`.

## Research rules

- Do not silently change a hypothesis. Update `3D_docs/hypothesis.md` and add a decision entry.
- Every experiment has a stable ID (`EXP-###`) and a record under `3D_docs/experiments/`.
- Record the question, protocol, split, configuration, metrics, result paths, interpretation, and conclusion.
- Never overwrite an earlier experiment record. Add a correction or a new experiment ID.
- Do not use held-out test data for model selection. The original six-episode test split was consumed by EXP-005 and is closed to further tuning.
- Distinguish controlled oracle evidence from deployable results. Known poses and frozen foundation output heads are not permitted to masquerade as online predictions.
- Store compact summaries and small machine-readable results in Git. Keep checkpoints, datasets, caches, and large generated artifacts out of Git.

## Current code map

- Active implementation: `revisit3d/`
- Historical probes: `skillmem/`, `ttt_continual/`, `dnpc/`
- Research source documents: `3D_docs/`
- Hypothesis retrospectives: `3D_docs/analysis/`
- Preserved long-form notes: `Research/`
- External reference repositories: `CUT3R/`, `FastVGGT/`, `Open-d4rt/`, `TTT3R/`, `UniSplat/`, `dust3r/`, `gaussian-splatting/`, `tttLRM/`, `vggt/`

Do not edit external reference repositories unless a task explicitly requires it.

## Runtime

Primary environment:

```bash
PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python <script>
```

Use development train/validation splits during iteration. Check GPU availability before launching a CUDA experiment.

## After completing a task

- Update `3D_docs/research_state.md`.
- Add or update the relevant `3D_docs/experiments/EXP-###_*.md` record.
- Add a `3D_docs/decisions.md` entry when a methodological choice changes.
- Link raw result files; do not replace the documented summary with an unstructured log dump.
- Report verification commands and known limitations.
