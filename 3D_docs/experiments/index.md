# Experiment Index

| ID | Question | Status | Main conclusion |
|---|---|---|---|
| [EXP-001](EXP-001_tttlrm_fastweight_premise.md) | Are tttLRM fast-weight updates reusable across physical revisits? | Completed | Raw reuse did not establish causal, context-selective benefit. |
| [EXP-002](EXP-002_revisit_benchmark_and_objective_health.md) | Is the benchmark and online geometry signal valid? | Completed | Cross-episode benchmark works; naïve reprojection was degenerate; frozen-track 3D consistency is usable for controlled probes. |
| [EXP-003](EXP-003_compact_state_reuse.md) | Can global/slot compact TTT states be reused? | Completed | Generic warm-start exists, but state-specific utility and retrieval selectivity fail. |
| [EXP-004](EXP-004_keys_and_update_routing.md) | Can learned keys and update compatibility retrieve the right experience? | Completed | Keys are weak and learned update vectors collapse; vector cosine is not an adequate reranker. |
| [EXP-005](EXP-005_dense_3d_atom_transport.md) | Does a spatially transported local atom show reusable causal utility? | Completed | Feasibility supported; learned utility/risk is required for safety. |
| [EXP-006](EXP-006_trainable_3d_atom_risk_routing.md) | Can local TTT memory and observable utility/risk routing generalize without oracle poses? | In progress; train OOF architecture gate complete | Visual local reuse is causal and reusable; predicted geometry is evidence rather than carrier; utility routing is feasible, but grouped risk supervision requires more train components. |

Long-form chronological analyses remain under `Research/` and are linked from the individual experiment records.
