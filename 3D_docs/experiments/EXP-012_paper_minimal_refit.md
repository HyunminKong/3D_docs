# EXP-012 — Paper-Minimal Refit

Status: Stage 0 registered before execution

## Question

Can the useful local-plasticity phenomenon survive after removing the feasibility-stage auxiliary losses and learned fine router?

## Paper-minimal candidate

The candidate has two learned concepts:

1. an 8-D per-token local plasticity decoder with a frozen train-PCA visual transport key;
2. one linear future-utility score that will later serve as both the memory address and the zero-threshold reuse decision.

The online step is frozen from EXP-011: one 3D frozen-track loss, one step, `eta=0.0125`. A retrieved code is applied by the existing bounded residual `alpha=0.10`. Predicted-geometry transport, key contrastive loss, smoothness, code norm, harmful-code neutralization, center loss, neural risk, and a separate fine router are absent.

## Stage 0 — minimal atom

Train only the decoder with one equal-weight meta-objective:

\[
\mathcal L_{meta}=\tfrac12\left[F_q(z_t)+F_q(z_t+\alpha Tz_i)\right].
\]

`q` is a disjoint future/query segment used only for offline meta-training. The source `i` is the metadata-matched revisit. There are no auxiliary loss weights. Training length is fixed at three epochs and is not selected on validation.

Five-fold physical-component cross-fitting must show:

- component-mean current/base future loss below 1;
- matched reuse utility above 0.5%;
- matched harmful rate no greater than 20%;
- a positive component-bootstrap lower bound for matched-source utility minus the distant within-episode control.

Failure stops this minimal head. Success authorizes a source-entity-safe unified utility address on train only. Validation/test are not accessed in Stage 0.

## Files

- Config: `configs/EXP-012_minimal_atom_v10.yaml`
- Experiment module: `revisit3d/experiments/exp012_minimal.py`
- Trainer: `revisit3d/scripts/train_exp012_minimal_atom.py`
- Result: `revisit3d/results/EXP-012/stage0_minimal_atom_crossfit_train_v10.json`
