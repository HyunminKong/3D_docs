# EXP-032 — Post-Terminal Coverage Accounting Audit

Status: Completed; metadata/result audit only

## Question

Why did the 214-direction EXP-021 manifest produce 187 EXP-031 evaluation
rows, and was the registered minimum of 190 unique targets feasible under the
frozen evaluator?

## Protocol

Read only the locked manifest, converted timestamp metadata, and already
written EXP-031 result. Reconstruct the evaluator's unique-context and
write-after-predict event accounting exactly. Do not decode sensors, run the
model, recompute metrics, change the gate, or rerun EXP-031.

The audit reports directional target multiplicity, unique targets, targets
with a non-empty causal bank, metric-valid evaluated targets, and exclusions by
location.

## Files

- Config: `configs/EXP-032_terminal_coverage_accounting_v10.yaml`
- Script: `revisit3d/scripts/audit_exp032_terminal_coverage.py`
- Result: `revisit3d/results/EXP-032/terminal_coverage_accounting_v10.json`

## Decision boundary

This audit cannot convert the registered EXP-031 failure into a pass. It can
only distinguish a model/data failure from an accounting error and define the
claim language that transparently accompanies the immutable terminal result.

## Result

- 214 directional manifest episodes contained 188 unique target contexts;
- 26 target occurrences were repeated by graph-connected directional edges;
- one Singapore-Onenorth target was the first event in its location and had an
  empty causal bank;
- the maximum possible unique causal-target count was therefore 187;
- EXP-031 evaluated exactly those 187 targets, with zero post-eligibility
  metric exclusions, across all 29 components.

The registered 190-target minimum exceeded the evaluator's maximum possible
coverage and was infeasible before Stage 2. This is a protocol accounting
error, not missing LiDAR, selective target removal, or model failure.

## Conclusion

EXP-031 remains a registered-gate failure. Its method comparisons are still an
unbiased all-eligible-target evaluation and can be reported as qualified
terminal evidence, with the accounting error disclosed explicitly.
