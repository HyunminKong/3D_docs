# EXP-055 — Learned Conditional Tangent

Status: Registered; one train-only fit authorized
Purpose: Test whether a token-only conditioner learns EXP-054's spatial tangent
structure on a disjoint train scene

## Protocol

Use the exact EXP-053 partition and data seed: 48 four-frame anchors from
`pumpkin`, `heads`, and `chess` for fitting, and 16 anchors from `stairs` for a
single before/after audit. These are all EXP-051 train-role scenes. Validation
and both terminal partitions remain closed.

Freeze the official TTT3R model and the generic EXP-052 `8 -> 768` basis. Fit
only one zero-initialized, bias-free `768 -> 8` map of the detached final
decoder patch token. Its scale is `1 + tanh(A LN(h_n))`. Differentiate the sole
median-scale-aligned relative-3D metric through the identical one-step online
consistency update. Use exactly 48 AdamW steps at learning rate `1e-4`, zero
weight decay, no scheduler, auxiliary loss, coefficient, second step, basis
update, or other module.

The initial audit must reproduce EXP-053's generic-basis initial rows. The
capacity-matched learned-global control is the immutable EXP-053 result with
SHA-256 `93416fa41a5b9b823d05e89072d655adced4d5139ae0b688858879b852c09345`.

## Registered gates

- exact 48-fit/16-audit coverage, finite gradients, conditioner change, exact
  zero-code parity within `1e-5`, and exact reproduction of EXP-053 initial
  metric/online gains within `1e-8`;
- positive final online-consistency gain;
- final metric gain and final-minus-initial gain have positive 95% anchor-
  bootstrap intervals;
- final metric harm is at most 25% and no worse than initial;
- learned conditional gain beats EXP-053's learned global-basis gain with a
  positive paired 95% interval;
- no validation or terminal access.

Passing authorizes registration of one scene-disjoint validation experiment,
not automatic validation access. Failure stops this minimal token-only
conditioner without learning-rate, loss, budget, or input-feature tuning.

## Artifacts

- Config: `configs/EXP-055_learned_conditional_tangent_v10.yaml`
- Runner: `revisit3d/scripts/fit_exp055_learned_conditional_tangent.py`
- Result: `revisit3d/results/EXP-055/learned_conditional_tangent_v10.json`
- Conditional checkpoint (only on pass):
  `revisit3d/checkpoints/exp055_learned_conditional_tangent_v10.pt`
