# EXP-022 — Metric-Alignment Diagnosis

Status: Registered before execution

## Question

Did the terminal proxy-oriented atom lose metric geometry already on training
contexts, or did the final refit fail only when transferred to unseen validation
components? Is any loss caused mainly by the learned zero-code readout or by the
single online TTT step?

## Protocol

Use the existing 218 unique train targets/25 components and sparse-LiDAR query
evaluation. Do not train or select a model. Compare:

1. frozen foundation depth;
2. the metric-healthy EXP-011 reference current TTT result;
3. the EXP-015 head at zero local code;
4. the EXP-015 head after the frozen one-step `track3D` update.

For the final head, decompose `base → zero-head → current TTT`, and correlate
the disjoint-query track3D improvement with SILog, aligned AbsRel, and 3D EPE
improvement. Component bootstrap intervals quantify every comparison. Query
LiDAR and future loss remain evaluation labels and never enter online TTT.

This is a diagnosis, not another atom variant. Its registered gate checks only
coverage and exact reproduction of the existing foundation baseline.

## Files

- Config: `configs/EXP-022_metric_alignment_diagnosis_v10.yaml`
- Evaluator: `revisit3d/scripts/diagnose_exp022_metric_alignment.py`
- Result: `revisit3d/results/EXP-022/stage0_metric_alignment_diagnosis_train_v10.json`
