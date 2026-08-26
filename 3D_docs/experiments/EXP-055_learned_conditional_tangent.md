# EXP-055 — Learned Conditional Tangent

Status: Completed; registered gate failed and no checkpoint was created
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

## Result

The conditioner completed all 48 registered steps, changed by L2 `0.0679`, and
kept exact zero-code parity. Its final scale mean/std on the audit were
`0.9844/0.2220`, confirming that the module learned nontrivial spatial axis
weights. Online consistency still improved by `5.76e-5`.

The weights did not generalize metric alignment. Mean metric gain moved from
`-1.40e-6` to `-0.34e-6`, but the final CI was
`[-2.60e-6, 2.11e-6]` and the paired improvement CI was
`[-0.55e-6, 2.63e-6]`. Harm fell from 56.25% to 43.75% but remained above the
25% gate. The model also trailed EXP-053's learned global basis by `0.90e-6`,
CI `[-2.56e-6, 0.72e-6]`. `stairs/seq-01` was weakly positive, while
`stairs/seq-02` remained negative with 62.5% harm.

The exact EXP-053 initial-row reproduction guard missed its preregistered
`1e-8` tolerance (`7.72e-6` maximum gain difference). Since all substantive
metric gates independently fail, no rerun or guard correction is scientifically
warranted. No checkpoint was written; validation and terminal remain closed.

EXP-054/055 therefore show a capacity-observability gap: a label-derived local
mask is highly useful, but a current-token-only linear map cannot infer it.
