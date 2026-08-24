# AGENTS.md

## Project

Continual Test-Time Adaptation Memory for Streaming 3D/4D Reconstruction.

This workspace contains the active research implementation plus several external reference repositories. The repository documents, not chat history, are the source of truth.

## Before starting any task

Read, in order:

1. `docs/research_state.md`
2. `docs/hypothesis.md`
3. `docs/decisions.md`
4. `docs/experiments/index.md`

For method or architecture work also read `docs/method.md`. For literature work start from `docs/literature/index.md`.

## Research rules

- Do not silently change a hypothesis. Update `docs/hypothesis.md` and add a decision entry.
- Every experiment has a stable ID (`EXP-###`) and a record under `docs/experiments/`.
- Record the question, protocol, split, configuration, metrics, result paths, interpretation, and conclusion.
- Never overwrite an earlier experiment record. Add a correction or a new experiment ID.
- Do not use held-out test data for model selection. The original six-episode test split was consumed by EXP-005 and is closed to further tuning.
- Distinguish controlled oracle evidence from deployable results. Known poses and frozen foundation output heads are not permitted to masquerade as online predictions.
- Store compact summaries and small machine-readable results in Git. Keep checkpoints, datasets, caches, and large generated artifacts out of Git.

## Current code map

- Active implementation: `revisit3d/`
- Historical probes: `skillmem/`, `ttt_continual/`, `dnpc/`
- Research source documents: `docs/`
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

- Update `docs/research_state.md`.
- Add or update the relevant `docs/experiments/EXP-###_*.md` record.
- Add a `docs/decisions.md` entry when a methodological choice changes.
- Link raw result files; do not replace the documented summary with an unstructured log dump.
- Report verification commands and known limitations.
