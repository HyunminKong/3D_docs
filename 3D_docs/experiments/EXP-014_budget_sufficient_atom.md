# EXP-014 — Optimization-Budget Sufficiency

Status: Registered before execution

## Question

Did the minimal frozen-key atom fail because its component folds received only 483–576 updates, below the 1000-step budget selected before EXP-012?

## Pre-existing evidence

The immutable EXP-006 v2.7 cross-fit selected 1000 steps: oracle utility was 1.323% at 500 steps and 3.855% at 1000. EXP-012 fixed three epochs, which yielded only 483–576 steps per fold and approximately 0.47% oracle utility. This audit is justified by the earlier result, not selected from EXP-012 outcomes.

## Registered protocol

Restore the frozen-PCA key and the Stage-0C minimal ranking objective. Change no loss, architecture, inference hyperparameter, candidate construction, fold, or success threshold. Train every fold and the final refit for exactly 1000 updates with the existing optimizer settings.

The unchanged gate requires current/base below 1, oracle utility above 1%, positive mean candidate utility, harm no greater than 30%, and positive component-bootstrap oracle-minus-mean headroom. Failure ends the minimal refit and requires a paper-scope pivot; success authorizes the unified utility address.

## Files

- Config: `configs/EXP-014_budget_sufficient_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp014_budget_sufficient_atom.py`
- Result: `revisit3d/results/EXP-014/stage0_budget_sufficient_atom_train_v10.json`
