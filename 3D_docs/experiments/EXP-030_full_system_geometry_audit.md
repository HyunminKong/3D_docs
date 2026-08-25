# EXP-030 — Frozen Full-System Geometry Audit

Status: Registered; not yet executed

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
