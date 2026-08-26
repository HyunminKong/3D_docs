# EXP-061 — Gauge/Local Error Anatomy

## Status

Completed; registered gate failed.

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

## Result

- Exact coverage: 4 scenes, 16 contexts, 64 frames.
- Fit/evaluation overlap: zero by checkerboard construction.
- Context Sim(3) EPE: `0.104893`.
- Matching per-frame Sim(3) EPE: `0.101050`.
- Cyclic per-frame transform EPE: `0.109494`.
- Matching-frame gain over context: `0.003843`, stratified context-bootstrap
  95% CI `[0.002209, 0.005591]`.
- Matching-frame gain over cyclic: `0.008444`, CI
  `[0.005083, 0.011868]`.
- Aggregate gauge fraction: `3.6636%` versus the registered `15%` minimum.
- Top-confidence-quartile gauge fraction: `6.4706%` versus the registered `10%`
  minimum; the `stairs` high-confidence scene effect is `-7.38e-5`.
- Matching-frame gain is positive in all four scene aggregates, but only 50/64
  individual frames; the high-confidence gain is positive in 42/64 frames.
- Peak allocated GPU memory: 4.62 GiB.
- Result SHA-256:
  `feb46823342b5a6caa95d74d9e4842de735e45cd46fa2b2591f38135bf9e2e93`.

## Interpretation

The disjoint-pixel and cyclic controls show that time-varying gauge error is a
real phenomenon rather than transform overfitting. It is nonetheless a small
fraction of short-stream TTT3R error and does not persist across every scene
after native high-confidence selection. The result supports reporting native
and locally aligned reliability separately, but not building the next paper
around a new gauge head.

## Conclusion

H19 is rejected as the central paper hypothesis for this registered
carrier/protocol. No calibration/head fit or validation access is authorized.
