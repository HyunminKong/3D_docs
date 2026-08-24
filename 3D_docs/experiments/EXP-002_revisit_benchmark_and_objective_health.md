# EXP-002 — Revisit Benchmark and Objective Health

## Question

Can we evaluate reusable TTT on real physical revisits without leaking query supervision, and does the online objective provide a non-degenerate gradient?

## Protocol

- Build component-safe nuScenes `A → B → A'` episodes.
- Keep context and future query frames disjoint.
- Audit reprojection valid support, pose scale/gauge, track coverage, and gradient norms.

## Result

- The physical revisit dataset and split protocol were established.
- Naïve photometric reprojection admitted zero-support/tiny-pose degeneracy.
- Frozen VGGT camera/track priors plus track 3D consistency supplied a healthy controlled online signal without a depth target.

## Conclusion

Use track 3D consistency for pre-framework probes. Do not claim the frozen foundation tracker/camera as the final deployable head.

## Sources

- `revisit3d/manifests/nuscenes_revisit_dev.json`
- `revisit3d/results/reprojection_geometry_val.json`
- `revisit3d/results/track_ttt_signal_val.json`
- `Research/pre_framework_validation_round2_2026-08-24.md`
