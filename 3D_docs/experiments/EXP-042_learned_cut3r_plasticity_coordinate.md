# EXP-042 — Learned CUT3R Plasticity Coordinate

Status: Completed; registered gate failed
Purpose: Scene-disjoint train-only oracle premise

## Question

Can the existing 6,144-parameter shared `8 -> 768` residual basis be learned
offline so that the same one-step online consistency gradients remain useful
for the current frame and become compatible across a physical revisit?

## Fixed source-safe split

Only the EXP-039 training manifest is available. In its frozen scene order,
the first 32 scenes supply four pairs each for fitting (128 pairs), and the next
16 disjoint scenes supply four pairs each for a single internal audit (64
pairs). The pair-ID hashes are respectively
`1682e3cec00a5b50f83553ccee2544ca23a4e9695a31dd92ff43da2e61a2b510`
and
`1b477d106c24cb1b22db76a9b8acd85e68af847b0a0a0ade7d88f26ed15c8992`.
Validation and terminal data remain closed.

## Frozen method and training budget

- Official CUT3R recurrence, encoder/decoder, pose memory, and DPT head are
  frozen.
- The only fitted tensor is the existing bias-free `8 -> 768` projection
  (6,144 parameters), initialized exactly as EXP-038--041.
- Online code creation remains one normalized `0.001` step on the one symmetric
  canonical-point consistency loss.
- Source-to-target transport is the already implemented frozen-token soft
  visual transport at temperature `0.07`.
- Offline fitting is one deterministic pass of 128 AdamW steps at learning rate
  `1e-4` and zero weight decay. There is no sweep, early stopping, scheduler,
  gradient clipping, or validation selection.
- Each offline step gives equal weight to the same target consistency loss
  after current-code application and after current plus oracle transported
  source-code application. Code-generation gradients are detached, making this
  a first-order meta-update. There is no alignment or auxiliary loss.

The audit compares the learned basis with its deterministic initial basis and
uses a deterministic spatial shuffle of the transported visual code as the
reuse control. No address, router, risk head, or memory bank is fit.

## Registered success gate

All of the following must hold on the 16-scene internal audit:

1. exact 128-pair fitting and 64-pair audit coverage, finite values, and a
   changed basis;
2. learned-basis current TTT improves over zero code;
3. learned-basis oracle visual reuse improves over current TTT;
4. full reuse beats its visual spatial shuffle;
5. transported source and target current codes have positive mean cosine
   agreement;
6. reuse harms at most 50% of pairs; and
7. learned-basis reuse gain exceeds the deterministic initial-basis gain on
   the same audit.

Failure ends the compact v2 memory direction defined by D119. Success freezes
this checkpoint and authorizes exactly one validation run before any utility
address is considered.

## Registered artifacts

- Config: `configs/EXP-042_learned_cut3r_plasticity_coordinate_v10.yaml`
- Script: `revisit3d/scripts/fit_exp042_learned_cut3r_plasticity_coordinate.py`
- Checkpoint (Git-ignored):
  `revisit3d/checkpoints/exp042_learned_cut3r_plasticity_coordinate_v10.pt`
- Result: `revisit3d/results/EXP-042/learned_cut3r_plasticity_coordinate_v10.json`

## Result

The fixed pass completed all 128 optimizer steps, changed the basis by L2
`0.625916`, and retained exact cached-readout parity (`0.0`). The checkpoint
SHA-256 is
`e9049ad104faca5303a4f29d089c9c100e4c329dd7e2a3943b9299b90582011d`.

| Internal-audit metric | Initial basis | Learned basis |
| --- | ---: | ---: |
| current TTT gain over zero code | `9.72e-5` | `3.59e-4` |
| oracle visual reuse gain over current | `4.24e-6` | `5.96e-6` |
| full gain over visual spatial shuffle | `3.52e-6` | `1.56e-6` |
| transported source/target code agreement | `0.00162` | `-0.00479` |
| reuse harm fraction | 51.56% | 46.88% |

The learned basis passed coverage, finiteness, basis-change, current-gain,
point-estimate reuse, shuffle, harm, and improvement-over-initial checks. It
failed the pre-registered positive mean code-agreement check, so the complete
gate failed.

A post-result uncertainty audit using 20,000 scene-level bootstrap resamples
with seed `4200010` further limits the point estimates. Learned current TTT gain
was positive in all 16 scenes with 95% interval
`[2.70e-4, 4.80e-4]`. Learned oracle reuse was positive in 10/16 scenes and
34/64 pairs, with interval `[-2.23e-5, 3.16e-5]`; its advantage over shuffle
also crossed zero, `[-1.95e-5, 2.37e-5]`. These intervals were not part of the
registered gate and are reported only to characterize the failed premise.

## Conclusion

Offline basis fitting clearly strengthens the one-step *current* adaptation
coordinate, but does not establish a stable revisit-compatible coordinate.
The mean reuse advantage is tiny and uncertain, code agreement is negative,
and nearly half of pairs are harmed. Under D119, this result does not authorize
validation access, a utility address, or a continual memory bank. The compact
competitive-carrier v2 direction stops here pending an explicit project-level
change of scope.
