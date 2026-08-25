# EXP-030 — Frozen Full-System Geometry Audit

Status: Completed; all registered gates passed

## Question

Does the frozen EXP-028 atom plus source-safe OOF EXP-029 address improve
SILog, aligned AbsRel, and 3D EPE directly, and beat matched random and
appearance policies?

## No-fit protocol

Recompute every causal candidate in the frozen EXP-029 panels. Address scores
are leave-one-location-out predictions with every held-location source removed.
Top-1 is applied only above semantic zero. Random and appearance use the same
panel and matched acceptance. No parameter, threshold, panel, or artifact is
changed; future RGB/LiDAR are evaluation readouts only.

## Registered gate

At least 200 targets and 20 physical components are required. The full policy
must improve the mean of all three primary metrics over current-only, matched
random, and appearance, with at least one positive component-bootstrap interval
in every comparison family. Failure keeps EXP-021 locked. Passing authorizes
assembly of one terminal artifact.

## Files

- Config: `configs/EXP-030_full_system_geometry_audit_v10.yaml`
- Runner: `revisit3d/scripts/evaluate_exp030_full_system_geometry.py`
- Result: `revisit3d/results/EXP-030/stage0_full_system_geometry_audit_v10.json`

## Result

The replay covered 217 targets and all 25 physical components. Acceptance was
the frozen EXP-029 semantic-zero policy. All primary comparisons passed.

| Policy | SILog | aligned AbsRel | 3D EPE (m) |
|---|---:|---:|---:|
| current-only | 52.8968 | 0.79680 | 6.42986 |
| full metric memory | 52.7704 | 0.79367 | 6.39500 |
| matched random | 52.8486 | 0.79567 | 6.41611 |
| appearance | 52.8524 | 0.79580 | 6.41673 |
| panel oracle | 52.6982 | 0.79236 | 6.37624 |

Full-minus-current improvements were 0.1265 SILog, 0.00314 AbsRel, and
0.03486 m EPE, with positive intervals for all three. Full also beat matched
random and appearance on all three means and all six component intervals.

## Conclusion

The frozen end-to-end development system is broadly metric healthy and its
benefit cannot be explained by generic memory warm-start or appearance
retrieval. No fit or threshold change occurred. The complete artifact may now
be assembled for one EXP-021 terminal evaluation.
