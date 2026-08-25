# EXP-015 — Final Core Utility Atom

Status: Registered before execution

## Question

Are absolute reuse quality and reuse-versus-current ranking jointly necessary to learn reusable local codes without feasibility-stage auxiliary losses?

## Factorial motivation

- EXP-012 Stage 0B used absolute current/reuse quality without ranking and failed.
- EXP-014 used current quality plus ranking at the pre-selected 1000-step budget; it reached 0.9205% oracle utility but missed the unchanged 1% gate.

## Final registered atom objective

Use the same single 3D-track signal to define

\[
\mathcal L_{meta}=\ell_{current}+\ell_{best}+\operatorname{softplus}(\ell_{best}-\operatorname{sg}[\ell_{current}]).
\]

There are no loss weights, key contrastive loss, neutralization loss, center loss, smoothness, or code norm. Architecture, frozen PCA key, candidates, 1000-step budget, folds, and all EXP-014 gates are unchanged. This is the final atom variant; failure ends the memory-paper path, while success freezes the head before fitting one unified utility address.

## Files

- Config: `configs/EXP-015_core_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp015_core_atom.py`
- Result: `revisit3d/results/EXP-015/stage0_core_atom_train_v10.json`
