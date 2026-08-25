# EXP-048 — FIFO Memory versus Second Current TTT

Status: Registered decisive novelty audit
Purpose: Test whether bounded memory is more than another optimization step

## Question

Does FIFO-16 agreement memory improve future consistency beyond spending the
same additional head/gradient budget on a second normalized current-code step?

## Protocol

Repeat the immutable 4,532-frame EXP-047 development streams. At every query,
compute one extra current TTT step from the first current code using the same
loss and normalized `0.001` step. Compare it with the frozen FIFO-16 agreement
memory, reservoir-16 appearance, and reservoir-16 random controls. Banks,
continuous recurrence, predict-before-write order, and all seeds remain fixed.

## Registered gate

FIFO agreement must have positive 95% scene-bootstrap gain over one-step
current, must beat the two-step current baseline with a positive lower bound,
and must beat appearance/random with positive lower bounds. FIFO harm must stay
below 10%, exact coverage/capacity/parity are required, and terminal remains
closed.

Failure means the active full-stream result is better explained as cached local
optimization than continual adaptation recall; terminal memory claims stop.

## Artifacts

- Config: `configs/EXP-048_fifo_memory_vs_second_ttt_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp047_full_stream_bounded_bank.py`
- Result: `revisit3d/results/EXP-048/fifo_memory_vs_second_ttt_v10.json`
