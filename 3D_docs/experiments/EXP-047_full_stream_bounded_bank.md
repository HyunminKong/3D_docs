# EXP-047 — Full-Stream Bounded Agreement Bank

Status: Registered development experiment
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
