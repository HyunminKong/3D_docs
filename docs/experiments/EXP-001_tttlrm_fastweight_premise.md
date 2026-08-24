# EXP-001 — tttLRM Fast-Weight Premise

## Question

Do updates produced by an existing TTT reconstruction model contain reusable, context-specific information across physical revisits?

## Protocol

Instrument tttLRM-style fast weights, compare matched/intervening/foreign updates, inspect low-rank structure, and inject past updates under oracle pairing.

## Result

Updates showed structure, but direct reuse did not produce a robust matched-over-foreign causal benefit. This ruled out treating an existing tttLRM fast-weight state as the proposed framework.

## Conclusion

Negative for the initial implementation. Start an independent model with a deliberately designed fast state.

## Sources

- `Research/pre_framework_hypothesis_tests_2026-08-20.md`
- `Research/revisit3d_preframework_results_2026-08-20.md`
