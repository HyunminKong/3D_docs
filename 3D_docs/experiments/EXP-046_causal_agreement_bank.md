# EXP-046 — Causal Agreement-Addressed Bank

Status: Registered development experiment
Purpose: Replace manifest pair identity with deployable candidate selection

## Question

Can the frozen current/memory agreement signal retrieve a useful record from a
causal multi-candidate bank without pose or physical-pair identity?

## Protocol

EXP-045 validation is now exposed development data. For every scene, construct
at most 16 unique source records using only `source_previous -> source`. A
record is eligible for a query only when its source frame index is strictly
earlier than the target. Pose and manifest pair identity are never selection
features.

Compare three parameter-free top-1 policies over the identical eligible bank:

- maximum transported-code agreement with the current code;
- maximum pooled frozen-token appearance cosine; and
- deterministic matched random.

Every selected candidate is independently transported and applied only when
its own agreement is positive. The frozen EXP-043 basis, one loss/step, visual
transport, and zero threshold are unchanged. The manifest-paired gated record
is reported only as a supplied-candidate reference.

## Registered development gate

All 213 queries/14 scenes and at least two causal candidates per query are
required. With 20,000 scene-bootstrap draws, agreement addressing must have a
positive gain over current and positive advantages over appearance and random.
Harm must not exceed 10%, and no scene may store over 16 records. Terminal
remains unopened.

Passing selects a parameter-free bank policy for later locked evaluation; it is
development evidence and not a new held-out claim.

## Registered artifacts

- Config: `configs/EXP-046_causal_agreement_bank_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp046_causal_agreement_bank.py`
- Result: `revisit3d/results/EXP-046/causal_agreement_bank_v10.json`
