# EXP-036 — Matched CUT3R and TTT3R TUM Baselines

Status: Registered before external-model evaluation

## Question

How do the official CUT3R recurrent update and TTT3R relevance-weighted update
perform on the same causal TUM event/query protocol used by EXP-035?

## Frozen protocol

Use the official final 512-DPT checkpoint, SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`,
through the TTT3R implementation's `cut3r` and `ttt3r` modes. For each sequence,
feed the exact eight context images of every EXP-034 event in timestamp order.
For registered revisit targets, append the four query images with
`update=false`; they produce a prediction but cannot change recurrent state.
Reset only between sequences.

Evaluate the self-view pointmap z-depth on the same 16×16 dense-depth cells and
the same SILog, median-aligned AbsRel, and 3D EPE definitions as EXP-035.
CUT3R/TTT3R receive their official 512-long-side preprocessing; Revisit3D uses
224×224 as frozen. This favors the external baselines in input resolution but
does not make their architecture or training budget matched.

No checkpoint, state rule, reset interval, confidence threshold, crop, frame
subsampling, or post-result setting may change. tttLRM is excluded from this
table because its released inference requires calibrated camera trajectories
and renders a scene representation rather than producing the same causal
query-read-only pointmap output; it remains a conceptual baseline.

## Gate

This is a reporting comparison, not model selection. Both modes must produce
finite metrics for all 111 targets and all three sequences. There is no
registered requirement that Revisit3D win; any result narrows paper language
but cannot alter the final model.

## Files

- Config: `configs/EXP-036_cut3r_ttt3r_baselines_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp036_cut3r_ttt3r_baselines.py`
- Result: `revisit3d/results/EXP-036/cut3r_ttt3r_tum_baselines_v10.json`
