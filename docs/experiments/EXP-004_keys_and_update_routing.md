# EXP-004 — Keys and Update Routing

## Question

Can appearance/geometry context keys retrieve a useful update, and can current-update cosine disambiguate candidates?

## Result

- Global context key: top-1 `0/14`, mean matched rank `5.36`.
- Learned local-token key: top-1 `2/14`, recall@3 `6/14`, recall@5 `7/14` for designated pairs.
- Learned update compatibility reranking: top-1 `0/14`; positive and negative cosine were both approximately one.
- Raw update cosine had a positive–negative gap (`0.9121` vs `0.8554`) but still selected no top-1 designated revisit.

## Interpretation

The retrieval key is weak, but the more fundamental failure is that a vector update is not a sufficiently discriminative or causally useful memory value. Future retrieval must be evaluated by utility regret, not only pair identity.

## Sources

- `revisit3d/results/context_key_dev_val.json`
- `revisit3d/results/local_key_retrieval_dev_val.json`
- `revisit3d/results/key_update_rerank_dev_val.json`
- `revisit3d/results/raw_update_rerank_dev_val.json`
