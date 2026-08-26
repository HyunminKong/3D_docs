# EXP-049 — Low-Parallax Complementary-Memory Oracle Premise

Status: Registered metadata stage; model gate pending
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
- Model config/result: registered after the metadata artifact is frozen
