# EXP-045 — Frozen Zero-Agreement Validation

Status: Completed; all registered gates passed
Purpose: First confirmatory test of parameter-free memory routing

## Question

Does the EXP-044 positive-agreement rule generalize without fitting to the
scene-disjoint EXP-039 validation split?

## Frozen candidate

- official frozen CUT3R carrier and DPT head;
- frozen EXP-043 exact-meta 6,144-parameter basis, SHA-256
  `eaa1c57f34cdb485099ba1e90cbd212c7d0243f725dcf3165015e2eec054a3a2`;
- one 8-D local-code step at `0.001` on the single symmetric consistency loss;
- frozen-token visual transport at temperature `0.07`;
- apply memory iff mean cosine with the current code is strictly greater than
  zero.

No parameter, threshold, loss, or module is fitted in EXP-045. All 213 pairs in
14 validation scenes are evaluated once. Terminal remains closed.

## Controls and gate

Compare zero code, current TTT, unconditional correct reuse, zero-agreement
correct reuse, and a spatially shuffled transported code routed by its own
agreement. With 20,000 scene-bootstrap draws, every condition below is
required:

1. exact coverage, finite values, and exact cached-readout parity;
2. positive 95% lower bound for current TTT over zero code;
3. positive lower bound for gated correct reuse over current;
4. positive lower bound for gated correct reuse over unconditional reuse;
5. positive lower bound for gated correct reuse over independently gated
   spatial shuffle;
6. routed harm at most 10%; and
7. nondegenerate 20--80% acceptance.

Failure stops this zero-agreement candidate. Success supports the utility-gate
premise but does not yet establish bank retrieval, continual capacity, or
absolute GT geometry improvement.

## Registered artifacts

- Config: `configs/EXP-045_zero_agreement_validation_v10.yaml`
- Script: `revisit3d/scripts/evaluate_exp045_zero_agreement_validation.py`
- Result: `revisit3d/results/EXP-045/zero_agreement_validation_v10.json`

## Result

All 213 pairs and 14 scenes were evaluated with exact cached-readout parity.

| Comparison | Scene-balanced gain | 95% CI | Positive scenes |
| --- | ---: | ---: | ---: |
| current TTT over zero code | `6.34e-4` | `[4.12e-4, 8.80e-4]` | 14/14 |
| gated correct reuse over current | `7.35e-5` | `[4.59e-5, 1.05e-4]` | 14/14 |
| gated correct over ungated | `6.68e-5` | `[3.21e-5, 1.13e-4]` | 14/14 |
| gated correct over gated shuffle | `5.29e-5` | `[3.22e-5, 7.54e-5]` | 14/14 |

Correct and shuffled candidates were each accepted on exactly 53.52% of pairs,
so the correct-over-shuffle result is not an acceptance-rate artifact. Gated
correct reuse harmed 3.76%, below the registered 10% maximum. Every gate passed.

## Conclusion

H10 is supported on untouched scene-disjoint validation. A parameter-free
descent-agreement sign test converts a harmful/uncertain raw memory correction
into a consistent future-utility gain without a router or tuned threshold. The
result validates routing for a supplied physical-revisit candidate; it does not
yet establish how that candidate is retrieved from a causal bank.
