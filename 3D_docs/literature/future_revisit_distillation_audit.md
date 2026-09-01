# Future-Revisit Distillation Collision Audit

Last audited: 2026-09-01

## Question

Is it still distinct to use later observations as an offline teacher for a
causal streaming 3D reconstruction prediction?

## Direct and adjacent prior art

| Work | Occupied capability | Boundary for this project |
|---|---|---|
| [CUT3R, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html) | A recurrent persistent state can reread/revisit image views after more context; the official evaluator repeats a sequence and scores the final pass. | Revisit inference and a persistent state are not novel. |
| [Point4Cast, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Point4Cast_Streaming_Dynamic_Scene_Reconstruction_and_Forecasting_CVPR_2026_paper.html) | A later streaming state reads pointmaps for past, present, and future query times and is trained with 3D supervision. | Past-time readout from a future-conditioned state is occupied as an architecture. |
| [FTKD, 2025/2026](https://arxiv.org/abs/2512.08247) | An offline future-frame teacher transfers future temporal knowledge to an online 3D detector with no inference overhead. | Generic future-to-causal knowledge distillation is occupied. |
| [4RC / S-4RC, ICML 2026](https://arxiv.org/abs/2602.10094) | Encode once and query geometry/motion at arbitrary source views and target times, including a causal streaming supplement. | Generic future-conditioned 4D querying is occupied. |
| [Context-Matched Distillation, 2026](https://arxiv.org/abs/2608.13391) | Shows that a bidirectional teacher can be ill matched to a causal student's information set in autoregressive video distillation. | A future teacher is not automatically valid; its correction must be empirically beneficial and causally learnable. |

## Remaining narrow premise

The unclaimed object tested here is not a new recurrent state, revisit decoder,
or generic temporal KD framework. It is the **same frozen streaming geometry
carrier's gauge-normalized future correction field**:

\[
\Delta_T(I_t,H_t,H_{t+1:T}) =
\bar P(I_t,S_T)-\bar P(I_t,S_t),
\]

where the query RGB is identical, neither readout writes state, and bars remove
the monocular scale gauge. Before any distillation method is allowed, EXP-071
asks whether this field (a) improves absolute 3D geometry beyond rereading the
target from its prefix state and (b) points toward the offline metric residual.

If both hold, a later experiment may test one source-safe self-distillation
loss on unlabeled video with unchanged causal inference. The permissible claim
would be geometry-specific **future correction amortization**, not novelty of
future knowledge distillation. Point4Cast must remain a central baseline and
the difference from its future-state past readout must be explicit: the
deployed model receives no future frames and performs no retroactive readout.

## Stop boundary

Failure of EXP-071 closes this direction without a new decoder, confidence
head, frame search, longer-context search, validation access, or distillation
fit. Passing only authorizes a causal-predictability/capacity experiment; it
does not establish a paper method.
