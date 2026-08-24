# Continual TTT Memory for Streaming 3D/4D Reconstruction

This repository is the source of truth for a PhD/CVPR/ICLR research project on reusable test-time adaptation in streaming 3D/4D reconstruction.

The current result is not a wrapper around tttLRM. The selected direction is a new geometry model built on frozen foundation features, with spatially addressable 3D plasticity atoms, geometry-only local TTT, correspondence-aware transport, and a learned future-utility/risk selector.

## Start here

- Current snapshot: [`docs/research_state.md`](docs/research_state.md)
- Hypotheses: [`docs/hypothesis.md`](docs/hypothesis.md)
- Current method: [`docs/method.md`](docs/method.md)
- Decisions: [`docs/decisions.md`](docs/decisions.md)
- Experiment index: [`docs/experiments/index.md`](docs/experiments/index.md)
- Literature index: [`docs/literature/index.md`](docs/literature/index.md)
- External dependency snapshot: [`docs/dependencies.md`](docs/dependencies.md)

## Repository layout

```text
AGENTS.md                    agent workflow and research rules
docs/                        official research state and experiment records
Research/                    preserved long-form analyses and earlier notes
revisit3d/                   active implementation and lightweight raw results
skillmem/, ttt_continual/    historical probes
dnpc/                        historical code; generated outputs are ignored
FastVGGT/, Open-d4rt/, ...   external reference repositories, ignored by this repo
```

## Working loop

```text
ChatGPT research discussion
        ↓
docs: hypothesis / method / decisions
        ↓
Codex implementation and experiments
        ↓
revisit3d code + results
        ↓
docs: experiment record + research_state update
        ↓
Git commit / GitHub
```

Chat history is not authoritative. A research change becomes official only after it is reflected in the repository documents and committed.

## Quick verification

```bash
PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python revisit3d/scripts/smoke_test.py
PYTHONPATH=. /home/khm/anaconda3/envs/tttlrm/bin/python revisit3d/scripts/meta_smoke_test.py
```

See [`revisit3d/README.md`](revisit3d/README.md) for implementation-specific commands. Checkpoints and datasets are intentionally not tracked.
