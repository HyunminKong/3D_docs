# EXP-033 — Frozen Paper-Model Efficiency and Complexity Audit

Status: Registered before profiling

## Question

What inference-time latency, accelerator memory, persistent bank storage, and
search scaling does the exact frozen EXP-028/029/031 candidate require?

## Protocol

Use one fixed development-train context at eight 224×224 views. Do not fit,
tune, or alter any model value. Profile:

1. frozen FastVGGT feature/geometry and tracker passes separately;
2. local atom construction, one current `track3D` step, visual transport, and
   residual depth readout from cached foundation outputs;
3. the exact factorized Ridge address at bank sizes 8–4096;
4. actual float32 plasticity-record bytes and reservoir-64 storage;
5. trainable parameter count and incremental peak CUDA allocation.

CUDA timings exclude dataset disk I/O and one-time model loading. Report warmup,
repetition counts, median, p90, and mean. The vectorized address is algebraically
identical to the scalar terminal score and must pass a numerical equivalence
check. Results characterize the implementation; they cannot authorize model
compression or architectural changes.

## Files

- Config: `configs/EXP-033_frozen_efficiency_audit_v10.yaml`
- Script: `revisit3d/scripts/profile_exp033_frozen_efficiency.py`
- Result: `revisit3d/results/EXP-033/frozen_efficiency_audit_v10.json`

## Gate

This is a reporting audit, not model selection. It is complete only if all
frozen artifact hashes match, every timing section is finite, address
factorization error is at most `1e-10`, and actual bank storage at capacity 64
is reported. No speed threshold is registered.
