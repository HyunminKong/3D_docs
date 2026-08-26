# EXP-061 — Gauge/Local Error Anatomy

## Status

Preregistered; not yet run.

## Question

Does frozen streaming TTT3R exhibit a time-varying, frame-shared Sim(3) error
that is distinct from local surface residual and persists among its most
confident predicted points?

## Hypothesis

H19. A single independent point confidence is incomplete when a shared gauge
latent creates correlated errors across the pointmap.

## Data contract

- 7Scenes train role frozen by EXP-051; validation and terminal roles unopened.
- Fresh sequences: `pumpkin/seq-02`, `heads/seq-02`, `chess/seq-02`,
  `stairs/seq-02`.
- Four fixed target indices per sequence: 79, 159, 239, 319.
- Four consecutive RGB-D frames ending at each target: 16 contexts, 64 frames.
- RGB alone enters frozen TTT3R. Depth, intrinsics, and pose are offline labels.

## Zero-fit protocol

1. Run official TTT3R recurrence independently on every four-frame context and
   read `pts3d_in_other_view` plus native cross-view confidence.
2. Back-project registered RGB-camera depth and transform it to the metric
   7Scenes world frame using ground-truth camera pose.
3. Define valid correspondences by same-image pixels, finite prediction/target,
   positive predicted depth, and target depth in `[0.2, 10]` metres.
4. Estimate transforms only on even checkerboard valid pixels. Deterministically
   cap each frame at 8192 fit correspondences so frames are equally weighted.
5. Fit (a) one Sim(3) to all four context frames and (b) one Sim(3) per frame by
   closed-form Umeyama alignment.
6. Score relative 3D EPE only on the disjoint odd checkerboard pixels.
7. Apply each per-frame transform to the next temporal frame as a cyclic control.
8. Repeat scoring on the top 25% native-confidence evaluation pixels in each
   frame. This selection uses no RGB-D error.

## Primary metrics

- `context_epe`: held-pixel error after one context Sim(3).
- `per_frame_epe`: held-pixel error after the matching frame Sim(3).
- `cyclic_epe`: held-pixel error after cyclic transform reassignment.
- `gauge_fraction = (context_epe - per_frame_epe) / context_epe`.
- The same four metrics on top-confidence-quartile pixels.
- Per-frame log-scale, rotation, and depth-normalized translation displacement
  relative to the context transform; native-confidence association is
  descriptive, not a gate.

## Frozen success gate

All of the following must hold:

1. exactly four scenes, 16 contexts, and 64 valid frames;
2. `context_epe - per_frame_epe > 0` in every scene and its context-bootstrap
   95% interval has positive lower bound;
3. `cyclic_epe - per_frame_epe > 0` in every scene and its context-bootstrap
   95% interval has positive lower bound;
4. aggregate `gauge_fraction >= 0.15`;
5. on top-confidence-quartile pixels, per-frame improvement is positive in
   every scene and aggregate `gauge_fraction >= 0.10`.

Success authorizes only a fresh observability experiment comparing predictors
of gauge risk and local risk. Failure closes H19 for this carrier/protocol; no
threshold or frame repair is allowed.

## Configuration and outputs

- Config: `configs/EXP-061_gauge_local_error_anatomy_v10.yaml`
- Preparation artifact: `revisit3d/results/EXP-061/selected_train_depth_registration_v10.json`
- Result: `revisit3d/results/EXP-061/gauge_local_error_anatomy_v10.json`
