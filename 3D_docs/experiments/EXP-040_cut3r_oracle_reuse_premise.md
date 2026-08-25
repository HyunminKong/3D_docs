# EXP-040 — CUT3R Local-Code Oracle Reuse Premise

Status: Registered before execution
Purpose: Train-only feasibility; no fitting and no validation/terminal access

## Question

With the accepted frozen CUT3R interface, does one local code step improve the
current frame, and does the correctly revisited source code contain additional
future-useful information after explicit 3D transport?

## Fixed protocol

- Open exactly four pairs from each of the first eight EXP-039 train-manifest
  scenes: 32 pairs total.
- Process `source-1 -> source -> target-1 -> target` with frozen CUT3R recurrence.
- At source and target, initialize the 8-D code to zero and take one normalized
  `0.001` step on the single symmetric canonical-point consistency loss.
- Transport the source code to target patches by nearest predicted canonical
  3D position and add it to the target current code with unit strength.
- Compare correct transport with a deterministic spatial shuffle of the same
  transported values.
- Fit neither the basis nor an address. Open no validation or terminal image.

## Registered gate

All of the following are required on scene-balanced train means:

1. source TTT lowers source consistency loss;
2. target current TTT lowers target consistency loss;
3. oracle source reuse improves over target current TTT;
4. correct spatial transport beats shuffled placement; and
5. oracle reuse harms at most 50% of individual pairs.

Passing authorizes train-only shared-basis meta-learning. Failure requires
revising the plasticity coordinate or online loss before an address or memory
bank is built.

## Artifacts

- Config: `configs/EXP-040_cut3r_oracle_reuse_premise_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp040_cut3r_oracle_reuse_premise.py`
- Result: `revisit3d/results/EXP-040/cut3r_oracle_reuse_premise_v10.json`
