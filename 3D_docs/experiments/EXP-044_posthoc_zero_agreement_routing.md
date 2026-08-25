# EXP-044 — Post-hoc Zero-Agreement Routing Diagnostic

Status: Registered as post-hoc analysis
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
