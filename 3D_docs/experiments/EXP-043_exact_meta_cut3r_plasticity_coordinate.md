# EXP-043 — Exact-Meta CUT3R Plasticity Coordinate

Status: Completed; ungated registered gate failed
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

## Result

After one external-process OOM before any artifact or audit access, the official
run was restarted from initialization and completed all 192 steps. Peak
allocated memory was 36.61 GiB, basis L2 change was `0.971665`, cached-readout
parity remained exact, and checkpoint SHA-256 is
`eaa1c57f34cdb485099ba1e90cbd212c7d0243f725dcf3165015e2eec054a3a2`.

| Audit metric | Initial basis | Exact-meta basis |
| --- | ---: | ---: |
| current TTT gain | `1.27e-4` | `6.42e-4` |
| current-gain 95% CI | `[7.67e-5, 1.92e-4]` | `[3.70e-4, 9.89e-4]` |
| ungated oracle reuse gain | `1.21e-6` | `9.30e-6` |
| reuse-gain 95% CI | `[-5.73e-6, 9.41e-6]` | `[-3.24e-5, 5.16e-5]` |
| full over visual shuffle | `1.53e-7` | `8.33e-8` |
| reuse harm | 58.33% | 50.00% |

The exact objective again learned a much stronger current update, positive in
all 15 scenes. Ungated reuse improved its point estimate over initialization,
but its interval crossed zero and correct visual transport was statistically
indistinguishable from spatial shuffle. The registered gate therefore failed
two checks and no direct validation run is authorized.

## Post-result diagnostic lead

The 60 learned-basis rows contain substantial heterogeneous utility: an oracle
current fallback would gain `6.12e-5`, versus `2.49e-5` for shuffled reuse. The
online-observable cosine agreement between transported source code and current
code correlates `0.752` with future gain. A zero-sign policy is analyzed
separately as post-hoc EXP-044; these inspected values are not EXP-043 evidence
and cannot be used to tune a threshold.

## Conclusion

H9 is rejected for *ungated* reuse. Exact differentiation solves current
plasticity but does not make every past code safe. The next scientifically
minimal question is whether the algebraic sign of current/memory descent
agreement supplies the missing parameter-free utility decision. This adds no
head, learned address, loss, or tuned scalar and must be frozen before any
validation access.
