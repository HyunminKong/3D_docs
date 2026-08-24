# EXP-003 — Compact State Reuse

## Question

Can a new global, slot-conditioned, or anchored compact TTT state preserve context-specific adaptation and improve future geometry?

## Protocol

Compare current-only, matched, intervening, foreign, and random reuse. Measure gradient cosine, centered singular spectrum, query utility, and convergence.

## Result

- A generic past learned update improved current-only by roughly `3.5e-3` in the controlled loss, but matched, intervening, and foreign updates were effectively indistinguishable.
- True 200-step meta-training made the update state more collapsed: matched–foreign utility gap was about `1.48e-7` and cosine values were essentially one.
- Raw anchored gradients retained variation but did not rank the causally best memory.

## Conclusion

Reject a small global/slot vector as the central memory object. Generic warm-start is not evidence of context-selective continual adaptation.

## Sources

- `revisit3d/results/learned_local_update_dev_200_val.json`
- `revisit3d/results/key_update_rerank_dev_200_val.json`
- `revisit3d/results/raw_update_utility_dev_val_scale1.json`
- `Research/pre_framework_validation_round2_2026-08-24.md`
