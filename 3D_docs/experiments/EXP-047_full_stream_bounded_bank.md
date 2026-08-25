# EXP-047 — Full-Stream Bounded Agreement Bank

Status: Completed; registered gate failed on reservoir versus FIFO
Purpose: Remove curated-write and pair-reset assumptions

## Question

Does agreement-addressed adaptation memory remain useful when every frame is
processed once in capture order and the bank stores ordinary stream frames
rather than manifest-selected revisit sources?

## Fixed protocol

Process 4,532 frames across the 14 now-exposed validation scenes, from each
scene's first frame through its last registered query. CUT3R recurrent state and
previous predicted geometry persist across the full stream. At each frame:

1. predict and create the one-step current code from the preceding frame;
2. retrieve and apply memory before writing the current record;
3. write the current code and frozen features to the bank.

The primary bank is a deterministic reservoir of exactly 16 records per scene.
Compare agreement addressing against appearance and random selection from the
same reservoir, plus agreement addressing from FIFO-16. Every policy uses its
selected candidate's own strict-positive agreement gate. There are no poses,
pair identities, learned routers, or threshold tuning.

## Registered development gate

All 4,532 stream frames and 213 unique queries must be processed. The
reservoir-agreement method must have positive 95% scene-bootstrap gain over
current and positive advantages over appearance, random, and FIFO agreement.
Harm must not exceed 10%, capacity must equal 16, and cached readout parity must
remain exact. Terminal remains unopened.

## Artifacts

- Config: `configs/EXP-047_full_stream_bounded_bank_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp047_full_stream_bounded_bank.py`
- Result: `revisit3d/results/EXP-047/full_stream_bounded_bank_v10.json`

## Result

Exact coverage and capacity were achieved. Both agreement banks are useful, but
the registered assumption that long-term reservoir retention beats FIFO was
reversed.

| Comparison | Gain | 95% CI | Positive scenes |
| --- | ---: | ---: | ---: |
| current TTT over zero code | `7.85e-4` | `[5.44e-4, 1.05e-3]` | 14/14 |
| reservoir agreement over current | `2.76e-4` | `[1.94e-4, 3.69e-4]` | 14/14 |
| reservoir over appearance | `1.87e-4` | `[1.16e-4, 2.70e-4]` | 14/14 |
| reservoir over random | `2.13e-4` | `[1.44e-4, 2.96e-4]` | 14/14 |
| reservoir over FIFO | `-6.24e-5` | `[-8.87e-5, -4.06e-5]` | 0/14 |

Reservoir agreement harms 0.47% and selects records 128.3 frames old on
average. FIFO agreement harms 0%, selects records only 6.18 frames old, and
gains `3.38e-4` over current with post-result descriptive CI
`[2.41e-4, 4.54e-4]`. The complete registered gate fails because FIFO is
strictly better in every scene.

## Conclusion

Every-frame bounded agreement memory works, and agreement—not appearance or
random selection—remains essential. But the result supports short-term FIFO
adaptation more strongly than long-term continual retention. Before terminal,
the method must beat an equal-budget second current TTT step; otherwise FIFO
memory may only cache a recent descent direction rather than recall reusable
long-term experience.
