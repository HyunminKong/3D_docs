# Experiment Index

| ID | Question | Status | Main conclusion |
|---|---|---|---|
| [EXP-001](EXP-001_tttlrm_fastweight_premise.md) | Are tttLRM fast-weight updates reusable across physical revisits? | Completed | Raw reuse did not establish causal, context-selective benefit. |
| [EXP-002](EXP-002_revisit_benchmark_and_objective_health.md) | Is the benchmark and online geometry signal valid? | Completed | Cross-episode benchmark works; naïve reprojection was degenerate; frozen-track 3D consistency is usable for controlled probes. |
| [EXP-003](EXP-003_compact_state_reuse.md) | Can global/slot compact TTT states be reused? | Completed | Generic warm-start exists, but state-specific utility and retrieval selectivity fail. |
| [EXP-004](EXP-004_keys_and_update_routing.md) | Can learned keys and update compatibility retrieve the right experience? | Completed | Keys are weak and learned update vectors collapse; vector cosine is not an adequate reranker. |
| [EXP-005](EXP-005_dense_3d_atom_transport.md) | Does a spatially transported local atom show reusable causal utility? | Completed | Feasibility supported; learned utility/risk is required for safety. |
| [EXP-006](EXP-006_trainable_3d_atom_risk_routing.md) | Can local TTT memory and observable utility/risk routing generalize without oracle poses? | Completed | One-shot descriptive validation supports visual local reuse and utility ranking; learned rejection and paper-scale generalization remain open. |
| [EXP-007](EXP-007_continual_atom_consolidation.md) | Can a causal, capacity-bounded atom bank retain routed revisit benefit? | Completed (partial) | Capacity-8 reuse is feasible and a separate frozen token key beats a matched permutation null; real-time order and safety remain open. |
| [EXP-008](EXP-008_true_timestamp_stream.md) | Does the dual-address bank survive unique writes in true capture-time order? | Completed (train) | The primary beat appearance diversity and 96.1% of matched compression nulls; independent unseen-scene generalization remains open. |
| [EXP-009](EXP-009_unseen_paperscale_benchmark.md) | Does utility-addressed local TTT memory generalize to fully unseen overlap components at paper scale? | Completed | The locked reservoir-64 system passed all terminal gates; it beat matched random addressing on 22 components, while reservoir superiority over FIFO was not established. |
| [EXP-010](EXP-010_paper_geometry_validity.md) | Does the locked utility improvement correspond to absolute depth/point accuracy? | Completed, gate failed | Aligned AbsRel improved, but SILog and 3D EPE worsened; broad geometry claim was rejected. |
| [EXP-011](EXP-011_objective_health.md) | Can a single self-supervised TTT loss improve all primary geometry metrics? | Stage 0 passed; validation registered | One 3D-track loss at eta 0.0125 was selected on train before one-shot validation. |

Long-form chronological analyses remain under `Research/` and are linked from the individual experiment records.
