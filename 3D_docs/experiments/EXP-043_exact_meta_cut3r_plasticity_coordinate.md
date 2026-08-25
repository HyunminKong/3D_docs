# EXP-043 — Exact-Meta CUT3R Plasticity Coordinate

Status: Registered; fit-only implementation smoke passed
Purpose: Approved minimal scope reopening after EXP-042

## Question

Does differentiating future geometry utility through the actual source and
target online code steps learn a revisit-compatible coordinate, without adding
an inference module or loss?

## Scope change

EXP-042 detached both generated codes before updating the basis. It therefore
optimized how a fixed code is decoded but did not optimize how the basis shapes
the TTT gradient that creates the code. EXP-043 retains the same CUT3R carrier,
6,144-parameter basis, 8-D local code, normalized step, consistency loss, visual
transport, and equal current/reuse objective. Its only methodological change is
the exact meta-gradient through both one-step code updates:

\[
z=-\eta\,\operatorname{RMSNorm}(\nabla_z L(B,z)|_{z=0}),
\qquad \nabla_B L_{future}(B,z(B)).
\]

This is standard differentiable optimization machinery rather than a separate
novelty claim. The paper hypothesis remains transported adaptation experience.

## Source-safe protocol

The 48 EXP-039 train scenes already exposed by EXP-042 become fit data (192
pairs). The remaining 15 previously unopened train scenes become the new
internal audit (60 pairs). Their pair-ID hashes are
`a07b9185db7000225a16a348d6d0d853356f26a27360e649ca87e35133811586`
and
`d2e5f742d0297f983325ef91c60387c1314db8518f860b73ce95914b3626dda9`.
Validation and terminal remain closed.

One technical smoke run may use only the first fit pair to verify second-order
autograd and memory feasibility. It may not change the registered learning
rate, loss, capacity, or audit protocol based on performance.

The first smoke exposed that this PyTorch build does not implement double
backward for its memory-efficient SDPA kernel. Re-running the identical
attention operation with PyTorch's math SDPA backend succeeded: the exact-meta
gradient was finite (`6.29e-4`), the basis changed, peak allocated CUDA memory
was 36.61 GiB, and no audit pair was accessed. This is an autograd backend
correction; it changes no model, objective, or registered hyperparameter.

## Fixed fit

- deterministic orthonormal initialization used by EXP-038--042;
- one pass, 192 AdamW steps, learning rate `1e-4`, zero weight decay;
- equal `0.5/0.5` weighting of current-code and current-plus-oracle-reuse
  instances of the same consistency loss;
- exact gradient through source and target code generation;
- no auxiliary/alignment loss, scheduler, clipping, sweep, early stopping,
  address, router, risk head, or memory bank.

## Registered success gate

On the 15-scene audit, all must hold:

1. exact coverage, finite values, nonzero basis change, and exact cached-readout
   parity;
2. the 95% scene-bootstrap lower bound is positive for current TTT gain;
3. the 95% scene-bootstrap lower bound is positive for oracle reuse over
   current;
4. the 95% scene-bootstrap lower bound is positive for full reuse over visual
   spatial shuffle;
5. reuse harms at most 50% of pairs; and
6. learned reuse point gain exceeds the deterministic initial-basis gain on the
   same audit.

Code cosine is diagnostic only: functional future utility, not coordinate
orientation, is the approved target. Success freezes the checkpoint and permits
one validation run. Failure ends this exact-meta realization before validation.

## Registered artifacts

- Config: `configs/EXP-043_exact_meta_cut3r_plasticity_coordinate_v10.yaml`
- Script: `revisit3d/scripts/fit_exp043_exact_meta_cut3r_plasticity_coordinate.py`
- Checkpoint (Git-ignored):
  `revisit3d/checkpoints/exp043_exact_meta_cut3r_plasticity_coordinate_v10.pt`
- Result:
  `revisit3d/results/EXP-043/exact_meta_cut3r_plasticity_coordinate_v10.json`
