# EXP-059 — EXP-058 Reproduction Accounting

Status: Completed; immutable-artifact audit only

## Question

Why did EXP-058 miss its EXP-057 second-current reproduction tolerance, and
does that mismatch invalidate the internally matched surface-fusion result?

## Protocol

Read only the immutable EXP-057 and EXP-058 JSON results after pinning their
SHA-256 hashes. Match all 16 rows by scene, sequence, and target frame. Compare
evaluation support and the erased, one-step, and two-step current errors. Do
not decode a frame, construct a model, rerun either experiment, change the
registered `1e-5` tolerance, recompute a metric, or alter either source result.

Report the maximum drift and its scale relative to EXP-058's already computed
within-run method margins. This comparison is accounting, not a new success
gate and not permission to relabel EXP-058.

## Decision boundary

EXP-058 remains a literal registered-gate failure regardless of this audit.
If row identity, support, or the pre-adaptation erased baseline differ, the
dependency result is not usable. If those quantities reproduce exactly and the
only mismatch is after repeated local-code differentiation, the internally
matched method comparisons may be retained as qualified positive evidence,
with the guard miss disclosed.

## Artifacts

- Config: `configs/EXP-059_exp058_reproduction_accounting_v10.yaml`
- Script: `revisit3d/scripts/audit_exp059_exp058_reproduction.py`
- Result: `revisit3d/results/EXP-059/exp058_reproduction_accounting_v10.json`

Result SHA-256:
`58cdb22bc2e79618e16a32b3079297c73280c4f54017ef65b9dc66b6cb77be93`.

## Result

- all 16 scene/sequence/frame identities and supported-pixel counts match;
- erased baseline maximum absolute difference is exactly zero;
- one-step and two-step maximum differences are `1.61976e-5` and
  `1.96695e-5` respectively;
- four second-current rows exceed the immutable `1e-5` guard;
- the maximum two-step drift is 0.00499% of EXP-058's mean fusion gain and
  0.01812% of its smallest scene-level fusion gain;
- all seven functional method checks in the immutable EXP-058 artifact pass.

## Conclusion

The mismatch is localized after local-code adaptation rather than to data,
support, or the frozen erased prediction. This does not repair EXP-058. It
supports reporting that experiment as qualified positive dependency evidence
whose internally matched effect is much larger than the observed repeated-step
drift.
