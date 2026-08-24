# EXP-005 — Dense 3D Atom Transport and Utility Routing

## Question

Does a spatially addressable local TTT state become causally reusable when transported by correspondence, and can current geometry evidence select useful past atoms?

## Protocol

- Per-token log-depth residual atom.
- One normalized local TTT update from track 3D consistency.
- Visual, oracle-world-coordinate, and combined coordinate/appearance transport.
- Candidate utility measured on disjoint future query frames.
- Candidate selection score sees only current context.

## Development result

| Method | matched - current | matched - foreign | designated utility top-1 |
|---|---:|---:|---:|
| Dense visual transport | `-2.29e-4` | `-3.30e-5` | `1/14` |
| Dense coordinate transport | `-3.36e-4` | `-9.69e-5` | `3/14` |
| Dense coordinate + appearance | `-3.76e-4` | `-1.06e-4` | `4/14` |

Full-bank online-score selection reduced mean future loss by `6.05e-4`. Visual-key top-5 plus online reranking reduced it by `4.51e-4`.

## Locked-test confirmation

Using the fixed coordinate+appearance probe on the original six episodes, online-score selection chose the oracle-best utility in `3/6` and reduced mean future loss by `7.34e-4`. It harmed `2/6` episodes, so heuristic selection is not safe enough.

## Conclusion

H1 and the oracle form of H2 are supported. H3 is partially supported. Implement a trainable 3D atom state and a future-utility/risk head; replace known-pose transport with predicted online geometry.

## Result files

- `revisit3d/results/dense_world_transport_probe_dev_val_r2_a5.json`
- `revisit3d/results/online_utility_proxy_dev_val.json`
- `revisit3d/results/local_key_online_rerank_dev_val_k5.json`
- `revisit3d/results/dense_world_transport_probe_test_r2_a5.json`
- `revisit3d/results/online_utility_proxy_test.json`

## Detailed analysis

- `Research/pre_framework_validation_round2_2026-08-24.md`
