# EXP-013 — Utility-Supervised Transport Key

Status: Completed; gate failed

## Question

Can direct future-utility supervision of the existing visual transport key recover reusable-code strength without adding an auxiliary loss or another module?

## Motivation

EXP-012 stopped the frozen-key compact family. Its strongest ranked variant preserved current TTT, positive candidate mean, acceptable harm, and significant selection headroom, but oracle reuse remained only 0.472%. A PCA key fixed independently of adaptation utility may be the bottleneck.

## Registered change

Use the exact EXP-012 Stage-0C architecture, five candidates, one online loss, relative-utility meta-objective, three epochs, cross-fit folds, and gates. Initialize the 64-D visual key from the same train PCA but allow that existing projection to receive gradients from the future utility objective. No contrastive key loss, geometry transport, regularizer, new head, or loss weight is added.

The unchanged gate requires current/base below 1, oracle utility above 1%, positive candidate-mean utility, harm no greater than 30%, and a positive component-bootstrap oracle-minus-mean interval. Failure ends local-key redesign for this paper. Success authorizes fitting one deployable unified utility address.

## Result

The trainable key increased selection headroom but overfit utility transport: current/base was `0.8010`, oracle utility `+0.521%`, mean candidate utility `-0.211%`, and harm `45.38%`. Oracle-minus-mean was significant (`+0.00732`, 95% CI `[+0.00515, +0.00977]`), but three registered gates failed. Utility-key redesign is stopped and no checkpoint was produced.

## Files

- Config: `configs/EXP-013_utility_key_atom_v10.yaml`
- Trainer: `revisit3d/scripts/train_exp013_utility_key_atom.py`
- Result: `revisit3d/results/EXP-013/stage0_utility_key_atom_train_v10.json`
