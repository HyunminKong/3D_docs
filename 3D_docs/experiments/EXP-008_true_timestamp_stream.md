# EXP-008 — True Capture-Time Continual Stream

Status: **Registered; Stage 0 not yet executed.**

## Question

Does the EXP-007 dual-address capacity-bounded atom bank retain its benefit when synthetic episode orders are replaced by actual nuScenes capture time and every unique context is written exactly once?

## Motivation

EXP-007 established causal memory mechanics but repeated the same contexts across ten pseudo-orders. It cannot show that consolidation survives a real observation order. Stage 0 is a correction using existing train-only OOF utility rows; it does not train or tune a new model.

## Locked Stage-0 protocol

- Use only the 76 expanded-train episodes and five component-safe OOF streams.
- Recover microsecond capture time from each converted CAM_FRONT filename.
- A context becomes available after its last context frame.
- Sort all unique contexts within each fold by capture time and context ID.
- Evaluate a target before writing its own context; write every context exactly once.
- Deduplicate repeated target events only after asserting that their candidate utility/router rows are identical.
- Query/future frames remain offline evaluation labels and never update online state.
- Compare unbounded unique, FIFO-8, appearance-diversity-8, oracle scene-latest-8, and strict crossfit frozen-bucket predicted-history-8.
- Do not access EXP-006 validation or the exposed test split.

## Registered success gate

The frozen-bucket predicted-history bank must:

1. exceed appearance-diversity mean utility;
2. not exceed its deadband harm;
3. retain at least 90% of oracle scene-latest utility.

Component bootstrap uncertainty is reported but is not a Stage-0 pass requirement because this is still a train-only feasibility correction.

## Config and output

- Config: `configs/EXP-008_true_timestamp_stream_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp008_true_timestamp_stream.py`
- Output: `revisit3d/results/EXP-008/stage0_true_timestamp_stream_train_v10.json`
