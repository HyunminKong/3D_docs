# EXP-017 — Adaptation-Context Utility Address

Status: Registered before execution

## Question

Does one observable self-improvement scalar supply the adaptation context needed for a unified utility address to beat matched random retrieval?

## Registered change

EXP-016's visual-only unified score passed six of seven gates but its random-difference CI crossed zero narrowly. For each context, append exactly one scalar

\[
h=(L_{pre}-L_{post})/(|L_{pre}|+\epsilon)
\]

computed by the already required current online step. The address descriptor becomes `[visual64,h]` and retains the same `[c,s,c-s,c*s]` factorization, now as exact 65-D MIPS. Ridge, alpha, source-safe folds, causal panels, top-1, zero threshold, comparators, and every gate remain unchanged. No head, router, loss, or inference hyperparameter is added.

Failure ends observable-address augmentation for the paper. Success freezes the 65-D all-train artifact for one-shot validation.

## Files

- Config: `configs/EXP-017_adaptation_context_address_v10.yaml`
- Fitter/evaluator: `revisit3d/scripts/fit_exp017_adaptation_context_address.py`
- Result: `revisit3d/results/EXP-017/stage0_adaptation_context_address_train_v10.json`
