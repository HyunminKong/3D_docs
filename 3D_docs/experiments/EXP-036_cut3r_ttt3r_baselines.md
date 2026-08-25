# EXP-036 — Matched CUT3R and TTT3R TUM Baselines

Status: Completed; reporting gate passed, competitiveness blocker found

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

## Result

Both official modes produced finite metrics for all 111 targets and all three
sequences. Sequence-balanced results were:

| Method | SILog ↓ | aligned AbsRel ↓ | 3D EPE ↓ |
|---|---:|---:|---:|
| Revisit3D current | 28.5429 | 0.231135 | 0.461097 m |
| Revisit3D full | 28.4625 | 0.230136 | 0.458924 m |
| CUT3R 512 | 16.6067 | 0.081164 | 0.239378 m |
| TTT3R 512 | **15.7271** | **0.078073** | **0.224615 m** |

TTT3R improves CUT3R on all three endpoints. Revisit3D full retains its
controlled within-backbone improvement but has 1.81× TTT3R SILog, 2.95× AbsRel,
and 2.04× 3D EPE. CUT3R and TTT3R process official 512-long-side inputs and are
trained as full reconstruction systems, whereas the current custom head uses
224×224 FastVGGT features. This is not a capacity-matched ablation, but the
absolute gap is too large to ignore.

CUT3R and TTT3R processed 2,228 event/query frames in 143.2 and 142.2 seconds
including preprocessing. Every registered reporting check passed.

The run used the `fastvggt` Python 3.10 environment with PyTorch 2.3.1,
Transformers 4.44.2, Roma 1.6.1, Accelerate 1.14.0, and OmegaConf 2.3.1. Two
pre-result attempts failed before a result could be written because of an
incompatible Transformers version and then a wrapper return-value unpack
error; neither changed the frozen protocol.

## Conclusion

The adaptation-memory mechanism is empirically real within its frozen
FastVGGT/custom-head control, but the current base reconstruction quality is not
competitive with released streaming systems. A broad CVPR claim of a new
state-of-the-art reconstruction framework is blocked. Proceeding requires
either integrating the compact plasticity-memory principle with a competitive
CUT3R/TTT3R-class backbone or explicitly narrowing the paper to a mechanism
study with a much weaker competitiveness claim.
