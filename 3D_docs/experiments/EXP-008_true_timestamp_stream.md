# EXP-008 — True Capture-Time Continual Stream

Status: **Completed on train; true-time feasibility supported.**

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

## Stage-0 result

Across 71 unique target contexts, the primary frozen-bucket predicted-history bank reached +0.02650 mean utility and 5.63% harm. Appearance-diversity reached +0.02387/7.04%, scene-latest +0.02486/5.63%, and unbounded unique +0.02339/7.04%. Primary minus appearance-diversity had a 19-component bootstrap mean +0.00244 with CI [+0.00019, +0.00486]. The registered gate passed.

The primary made 85 bucket merges, including 51 cross-scene diagnostic merges. Stage 1 therefore preserves each fold's OOF probability distribution but permutes scores among context pairs 1,000 times. The key association is supported only if observed utility exceeds 95% of this matched null while retaining no more harm than appearance diversity.

## Stage-1 result

The matched null had mean utility +0.02409 and 2.5/50/97.5 percentiles [+0.02155, +0.02407, +0.02662]. The observed +0.02650 result was at the 96.1st percentile (one-sided p=0.03996), and its 5.63% harm remained below appearance-diversity's 7.04%. The registered gate passed.

## Conclusion

The EXP-007 architecture choice survives the chronology correction. A dual-address capacity-bounded bank is therefore the selected static-revisit design:

- learned local key for token-level code transport;
- separate frozen token-set key for consolidation/prefiltering;
- past-only predicted utility history for retention;
- small utility-ranked candidate set;
- fixed 0.10 residual and current-only fallback for present damage control.

This is still train-only evidence derived from selected overlap episodes. It does not validate a final place encoder, universal capacity, bank-aware rejection, or paper-scale generalization. The next experiment must use scenes that were never part of EXP-001–008.

## Compact artifact

`revisit3d/results/EXP-008/summary_v11.json` records raw-result hashes and all decision metrics. Full event rows remain local and Git-ignored.
