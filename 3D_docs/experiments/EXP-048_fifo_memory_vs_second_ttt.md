# EXP-048 — FIFO Memory versus Second Current TTT

Status: Completed; decisive novelty gate failed
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

## Result

The immutable evaluator processed all 4,532 frames, 213 registered queries,
and 14 development scenes with exact cached-readout parity and capacity-16
banks. The first current step improved the scene-balanced consistency loss by
`7.8244e-4` (95% scene-bootstrap CI `[5.4371e-4, 1.0521e-3]`). A second current
step added `7.7791e-4` over the first step (CI `[5.4163e-4, 1.0402e-3]`).

FIFO-16 agreement memory also improved the first current step by `3.3759e-4`
(CI `[2.4012e-4, 4.5095e-4]`) and harmed one of 213 queries (0.47%). It beat
reservoir appearance and random addressing in all 14 scenes. However, it was
worse than the equal-step second-current baseline by `4.4033e-4`; the 95% CI
was wholly negative (`[-6.1767e-4, -2.8623e-4]`) and zero of 14 scenes favored
FIFO memory.

## Conclusion

The registered gate fails only on the decisive memory-specific comparison.
Recent FIFO records are useful cached update directions, but the result is
better explained by spending the same adaptation budget on the current
observation than by recalling past experience. The competitive-carrier branch
therefore does not support a continual-memory novelty claim and must not open
the terminal split. Any continuation requires a new project-level decision and
a materially different hypothesis, not a post-hoc threshold, capacity, or
retention repair on this exposed development set.
