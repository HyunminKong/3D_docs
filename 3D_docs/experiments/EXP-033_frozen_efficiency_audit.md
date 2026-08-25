# EXP-033 — Frozen Paper-Model Efficiency and Complexity Audit

Status: Completed; all reporting gates passed

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

## Result

All measurements use an A100-SXM4-80GB and one eight-view 224×224 development
context. Median latency was:

| Operation | Median latency |
|---|---:|
| frozen feature + custom base geometry | 102.983 ms |
| frozen geometry tracker | 189.345 ms |
| current-only local TTT after foundation | 1.750 ms |
| full local-memory path after foundation, excluding address | 1.994 ms |
| exact address search, bank 64 | 0.0020 ms CPU |
| exact address search, bank 4096 | 0.0231 ms CPU |

The measured full learned method adds approximately 1.996 ms to 292.328 ms of
separate frozen-foundation passes, or 0.68%. Memory reuse beyond current-only
TTT costs approximately 0.246 ms at capacity 64. These ratios exclude disk I/O,
host transfer, and one-time model loading and describe the current two-pass
foundation implementation.

The plasticity head plus factorized address contains 288,386 parameters; the
head occupies 1.10 MiB in float32. One actual float32 memory record has 631,040
bytes of tensor payload, and reservoir-64 stores 38.52 MiB excluding Python
container overhead. The 64-D per-token key accounts for 524,288 bytes (83.1%)
per record. Full method incremental peak CUDA allocation after the cached
foundation tensors are resident was 64.7 MiB.

Address factorization error was `5.20e-18`. All hashes, finite-timing,
factorization, and storage reporting checks passed.

## Conclusion

The proposed adaptation/retrieval computation is negligible relative to the
current frozen foundation implementation and search is not a bottleneck at the
registered capacity. Persistent token-key storage, not latency or learned
parameters, is the principal efficiency cost. No compression variant is
authorized by this audit.
