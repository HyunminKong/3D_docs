# EXP-044 — Post-hoc Zero-Agreement Routing Diagnostic

Status: Completed as post-hoc analysis
Purpose: Preserve the EXP-043 routing lead without confirmatory claims

## Question

Does the algebraic sign of current/memory code agreement explain the
heterogeneous future utility observed after EXP-043?

## Analysis contract

This analysis uses only the already exposed 60 learned-basis audit rows from
EXP-043, whose result SHA-256 is
`b26ac5231a2efe5e1d86773f426c29a1cfb1a31cec4a7c34fa1c3bc9ac8b3ee4`.
It compares current-only, unconditional reuse, reuse iff mean code cosine is
strictly positive, and oracle fallback. The threshold is exactly zero and is
not fitted. Uncertainty uses 20,000 scene bootstrap draws.

The result is explicitly post-hoc because the agreement policy was selected
after inspecting EXP-043. It cannot validate the method. Its only possible use
is to freeze a hypothesis for a later untouched split.

## Artifacts

- Config: `configs/EXP-044_posthoc_zero_agreement_routing_v10.yaml`
- Script: `revisit3d/scripts/analyze_exp044_posthoc_zero_agreement_routing.py`
- Result: `revisit3d/results/EXP-044/posthoc_zero_agreement_routing_v10.json`

## Result

Agreement has Pearson `r=0.7518` with future reuse utility. Unconditional reuse
gains `9.30e-6`, has CI `[-3.30e-5, 5.02e-5]`, and harms 50.0%. The fixed
positive-agreement rule accepts 48.33%, gains `6.05e-5` with scene-bootstrap CI
`[2.60e-5, 1.04e-4]`, improves 14/15 scenes, and harms 1.67%. Oracle fallback
gains `6.12e-5`; thus the post-hoc rule retains nearly all observed oracle
headroom on this exposed audit.

## Conclusion

The failed ungated mean hides a sharply addressable utility structure. This is
not confirmatory evidence, but it justifies freezing the algebraic zero rule for
one untouched validation evaluation. No threshold was fitted and no learned
router was introduced.
