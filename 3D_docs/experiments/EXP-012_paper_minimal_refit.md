# EXP-012 — Paper-Minimal Refit

Status: Stages 0A/0B failed; Stage 0C registered before execution

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

## Stage 0A result — matched identity is not utility

The decoder itself learned a strong current update: component-mean future current/base loss was `0.8304`. Matched reuse utility was only `+0.270%`, below the 0.5% gate, while the distant control reached `+0.312%`. Matched minus distant was `-0.00042`, 95% CI `[-0.00128, +0.00040]`. Harm was acceptable at 17.78%, but the full gate failed and no checkpoint was produced.

This rejects the training assumption that metadata-matched identity should be the positive adaptation. It does not reject utility-selected local reuse.

## Registered Stage 0B — utility-selected target

Keep exactly the same architecture, online loss, step size, residual, optimizer, fixed three epochs, and five component folds. The only change is the offline meta-label: from five train-only candidates (`matched`, `distant`, three deterministic foreign sources), select the lowest future track-loss candidate. The single meta-objective becomes

\[
\mathcal L_{meta}=\tfrac12\left[F_q(z_t)+\min_{i\in\mathcal C_t}F_q(z_t+\alpha Tz_i)\right].
\]

This is one loss with detached discrete candidate selection, not an added module or auxiliary term. It follows the established evidence that useful adaptation is defined by future utility rather than episode identity.

The OOF gate requires current/base below 1, oracle utility above 1%, candidate-mean utility above zero, candidate harm no greater than 30%, and a positive component-bootstrap lower bound for oracle-minus-candidate-mean headroom. Passing authorizes the unified utility scorer; failing stops this minimal atom family.

## Stage 0B result — selection exists, equal averaging is too weak

Current/base was `0.8352`, mean candidate utility was `+0.176%`, and oracle-minus-mean was `+0.00289`, 95% CI `[+0.00236, +0.00344]`. However, oracle utility was only `+0.465%` and candidate harm was `30.0275%`; both corresponding gates failed. No checkpoint was produced.

## Registered Stage 0C — unweighted relative-utility ranking

Keep Stage-0B data, candidates, architecture, and optimization unchanged, and preserve every Stage-0B threshold. Replace the equal average by the minimal relative-utility objective

\[
\mathcal L_{meta}=\ell_{current}+\operatorname{softplus}(\ell_{best}-\operatorname{sg}[\ell_{current}]),
\]

where both losses are normalized by frozen base future loss. This has two semantically necessary terms—absolute current quality and reuse-below-current ranking—no tuned loss weight, and no auxiliary loss. If the unchanged gate fails, this compact atom family is stopped.

## Files

- Config: `configs/EXP-012_minimal_atom_v10.yaml`
- Stage-0B config: `configs/EXP-012_utility_selected_atom_v11.yaml`
- Stage-0C config: `configs/EXP-012_ranked_atom_v12.yaml`
- Experiment module: `revisit3d/experiments/exp012_minimal.py`
- Trainer: `revisit3d/scripts/train_exp012_minimal_atom.py`
- Stage-0B trainer: `revisit3d/scripts/train_exp012_utility_selected_atom.py`
- Stage-0C trainer: `revisit3d/scripts/train_exp012_ranked_atom.py`
- Result: `revisit3d/results/EXP-012/stage0_minimal_atom_crossfit_train_v10.json`
- Stage-0B result: `revisit3d/results/EXP-012/stage0b_utility_selected_atom_train_v11.json`
- Stage-0C result: `revisit3d/results/EXP-012/stage0c_ranked_atom_train_v12.json`
