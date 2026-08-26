# EXP-051 — Fresh-Data and TTT3R Metric-Aligned Prerequisites

Status: Corrected v1.2 completed; all gates passed
Purpose: Freeze source-safe metric data and establish an exact step-wise TTT3R
carrier before designing a new current-only update

## Question

Can unused local 7Scenes data provide scene-disjoint train, validation, and
terminal roles with RGB/depth/pose triplets, and can the local plasticity
interface reproduce official TTT3R recurrent predictions at zero code?

## Registered protocol

Stage A is metadata-only. Order the seven scene names by SHA-256 of
`5100010:<scene>`, assign the first four to train, the next one to validation,
and the last two to terminal. Inventory file names and sizes without decoding
RGB, depth, or pose arrays and without loading a model. All sequences belonging
to a scene inherit the same role.

The expected assignment is train `pumpkin, heads, chess, stairs`, validation
`fire`, and terminal `office, redkitchen`. The terminal role is immutable and
must not be opened for method design or selection.

Stage B may decode only the first eight frames of train sequence
`pumpkin/seq-01`. Run official native TTT3R and the step-wise carrier at zero
code with the same RGB preprocessing and update/reset flags. Compare
`pts3d_in_self_view`, `pts3d_in_other_view`, and `camera_pose` frame by frame.
No basis is fit and no depth or pose ground truth is decoded in this parity
stage.

## Registered gates

Stage A passes only if all 43,000 expected frames form complete RGB/depth/pose
triplets across 46 sequences, the 4/1/2 scene roles are disjoint, role frame
counts equal 17,000/4,000/22,000, and no sensor/model access occurs.

Stage B passes only if maximum native/step-wise absolute output difference is at
most `1e-5` on all eight frames. Failure is an interface error and prohibits
metric-aligned training. Success authorizes a train-only zero-fit gradient and
one-step metric-utility premise; it does not authorize fitting or validation
access by itself.

## Fixed method boundary

The next candidate is current-only: official TTT3R recurrent state, one 8-D
per-token code, one online symmetric predicted-3D consistency loss, and one
normalized code step. Offline RGB-D labels may shape the shared code basis, but
there is no memory, retrieval, transport, router, extra inference head, or
online ground-truth access.

## Artifacts

- Config: `configs/EXP-051_ttt3r_metric_aligned_prerequisites_v10.yaml`
- Metadata audit: `revisit3d/scripts/audit_exp051_7scenes_source_safe_partition.py`
- TTT3R parity audit: `revisit3d/scripts/audit_exp051_ttt3r_stepwise_parity.py`
- Results: `revisit3d/results/EXP-051/`

## v1.0 parity correction note

The immutable v1.0 parity result matches native output exactly through frame 2
but diverges after the first soft TTT3R state update (`4.80e-3` maximum). The
manual tensor-layout transcription is therefore rejected. v1.1 changes only
that transcription to the exact official `einops.rearrange` expression. It
uses the same eight train RGB frames, model, flags, outputs, tolerance, and
registered gate. The v1.0 result remains preserved; no metric ground truth,
validation, or terminal input is accessed.

The v1.1 model comparison then achieves exact zero error on every output and
frame. Its aggregate `passed` field is nevertheless false because two factual
fields named `ground_truth_accessed` and `terminal_accessed` correctly contain
`false` but were incorrectly included directly in `all(checks.values())`.
v1.2 changes only those check names to the positive propositions
`no_ground_truth_access` and `no_terminal_access` and repeats the immutable
parity protocol. No threshold, input, or model operation changes.

## Result

The metadata stage passed every gate: 43,000 complete RGB/depth/pose triplets
across 46 sequences, with 17,000/4,000/22,000 frames in disjoint
train/validation/terminal roles. The frozen manifest hashes are
`f03232b8a36f09eeb869875c43c616bca1d4e9c0bc4477f8902bc31b01162f9a`,
`236d00ffcb764fc596e1e31fee3571461e1f0892c426b0a2639f3a92f3746eef`,
and `df056310644e2d9aefd48b538f6dca8ce94f9f02cabc1272c5e82fe631630ead`.

The corrected v1.2 step-wise carrier matches official native TTT3R exactly on
all eight registered train frames: maximum absolute difference is `0` for
self-view points, other-view points, and camera pose. No depth/pose ground
truth, validation, or terminal input was accessed. The train-only metric
alignment premise is authorized.
