# EXP-049 — Low-Parallax Complementary-Memory Oracle Premise

Status: Completed; oracle gate failed
Purpose: Test whether past adaptation has unique value when current geometry is
underdetermined by weak adjacent camera translation

## Question

On a competitive frozen CUT3R carrier, can a pose-supplied past adaptation from
a stronger-baseline source beat an equal normalized second current TTT step at
a low-parallax revisit, and is that advantage larger than in motion-sufficient
targets?

## Metadata-only definition

Use only the EXP-039 train manifest and its already public camera trajectory
metadata. Normalize adjacent translation by each scene's median nonzero step.
A complementary low-parallax pair has source adjacent translation at least
`1.0` and target adjacent translation at most `0.5`. Its within-scene control
has both source and target translation at least `1.0`. Keep only scenes with
both regimes and select the first manifest pair per regime. The registered
coverage is 24 scenes and 48 pairs.

This definition is fixed before image decoding. Pose is used only to construct
the offline oracle candidate and regime label; it is not a deployable retrieval
result. Validation and terminal remain closed.

## Frozen model comparison

Use the frozen EXP-043 exact-meta 8-D basis, one symmetric point-consistency
loss, normalized step `0.001`, visual transport, and the algebraic positive
agreement rule. Compare base, one current step, equal-size second current step,
supplied complementary memory, spatially shuffled memory, and future-oracle
fallback between current-only and supplied memory. No fitting or parameter
selection is allowed.

## Mandatory gate

The low-parallax future-oracle fallback must beat the second current step with a
positive scene-bootstrap lower bound. Its advantage over second current must
also be significantly larger than the corresponding motion-sufficient
advantage. The supplied agreement-gated memory, shuffle control, acceptance,
and harm are diagnostic; they cannot rescue failure of the oracle gate.

Failure means the existing local update code contains no demonstrated unique
past information even in the registered weak-parallax regime. It stops this
memory object before a new retrieval module or terminal access.

## Artifacts

- Metadata config: `configs/EXP-049_low_parallax_oracle_manifest_v10.yaml`
- Metadata builder: `revisit3d/scripts/build_exp049_low_parallax_oracle_manifest.py`
- Frozen pair manifest: `revisit3d/manifests/exp049_low_parallax_oracle_train_v10.json`
  (SHA-256 `88fff50e4641026c966ea3edc880f27381ceae87e2523cd73bdf42d34bcd4d82`)
- Metadata audit: `revisit3d/results/EXP-049/low_parallax_manifest_audit_v10.json`
- Model config: `configs/EXP-049_low_parallax_oracle_premise_v10.yaml`
- Evaluator: `revisit3d/scripts/evaluate_exp049_low_parallax_oracle_premise.py`
- Model result: `revisit3d/results/EXP-049/low_parallax_oracle_premise_v10.json`

## Result

The frozen evaluator completed all 48 train pairs from 24 scenes with exact
zero-code/readout parity and no validation or terminal access. In the
low-parallax regime, the first current step improved consistency by
`5.52e-4`, and the equal second current step added `5.78e-4` with CI
`[4.45e-4, 7.33e-4]`; all 24 scenes improved.

The supplied past code was future-useful over one-step current in 45.83% of
low-parallax pairs. Even an offline future-oracle fallback that chose the
better of one-step current and supplied memory was nevertheless worse than the
second current step by `5.18e-4`, CI `[-6.40e-4, -4.02e-4]`, with zero of 24
scenes favoring memory. Agreement-gated memory had the same failure. Raw memory
did not beat a spatial shuffle: the mean was `-8.40e-6`, CI
`[-6.89e-5, 5.82e-5]`.

The low-minus-sufficient oracle interaction was `-6.46e-5`, CI
`[-1.89e-4, 6.90e-5]`; it did not support a low-parallax-specific advantage.

## Conclusion

Both mandatory oracle gates fail. Weak adjacent camera translation alone does
not make the tested online consistency objective information-insufficient:
repeated current optimization remains effective in every scene. More
fundamentally, the existing 8-D adaptation-direction code does not carry
complementary geometry that can beat repeated current optimization, even with
offline pose pairing and future-oracle application. This memory object is
stopped; no router, bank, threshold, or terminal experiment is authorized from
EXP-049.
