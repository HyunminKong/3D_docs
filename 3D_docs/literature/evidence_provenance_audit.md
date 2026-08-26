# Causal Evidence-Provenance Audit for Streaming 3D (2023--2026)

Last audited: 2026-08-26

## Candidate problem

CUT3R can write an RGB history into a recurrent state and later answer a camera
ray query without query RGB. The returned pointmap can combine geometry
supported by earlier observations with completion inferred only from the model
prior. Native confidence reports belief but does not explicitly say whether a
point is grounded in causal observations. The candidate question is whether
this distinction is measurable and useful for risk ranking.

## Occupied directions

| Direction | Representative work | Boundary for this project |
|---|---|---|
| Generic pointmap uncertainty | Trust3R (2026) and confidence heads in DUSt3R-family models | A new generic uncertainty or calibration head is insufficient. |
| Rendering provenance | ProvNeRF (NeurIPS 2024) | Provenance is established for NeRF render evidence; novelty must be causal streaming pointmap evidence. |
| Explicit recurrent/map memory | Point3R, LONG3R, Mem3R | Storing another geometry memory is not the contribution. EXP-066 reads the existing state only. |
| Active reconstruction/view selection | AREA3D (CVPR 2026) | The claim cannot be active view planning or next-best-view selection. |
| Confidence calibration | conformal and post-hoc 3D confidence work | Threshold calibration alone is occupied and EXP-066 fits no threshold. |

## Remaining defensible boundary

The narrow unresolved object is an **observation-support provenance field for
an RGB-free ray-only query**:

1. write only past RGB views into the frozen recurrent carrier;
2. query a later camera using rays and `update=false`, without its RGB;
3. label query patches offline as geometrically visible or not visible in the
   causal history;
4. derive a source-safe predicted signal only from the query and prior
   predicted pointmaps; and
5. test whether it explains absolute geometry risk beyond native confidence.

The fixed premise signal is nearest predicted 3D distance from a query point to
the union of history points, normalized by predicted query range. It has no
learned weights or fitted threshold. GT poses, depths, and visibility are
offline labels only; the ray query uses registered camera pose solely as a
controlled query coordinate and is not a deployable camera-estimation result.

## Stop boundary

Failure ends this signal without feature engineering, distance variants, a
learned head, or validation access. Success establishes only the phenomenon and
authorizes a broader baseline/capacity decision. It does not authorize a new
memory bank, uncertainty network, active planner, or reliability claim.

## Primary sources

- CUT3R, CVPR 2025: <https://openaccess.thecvf.com/content/CVPR2025/html/Wang_CUT3R_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html>
- ProvNeRF, NeurIPS 2024: <https://proceedings.neurips.cc/paper_files/paper/2024/file/b3a08d179347e33414badadf100e4e8d-Paper-Conference.pdf>
- Point3R, NeurIPS 2025: <https://papers.neurips.cc/paper_files/paper/2025/file/650db8e1b0b016dc270d51c1476e91cf-Paper-Conference.pdf>
- Trust3R: <https://arxiv.org/abs/2605.19539>
- AREA3D, CVPR 2026: <https://openaccess.thecvf.com/content/CVPR2026/html/Xu_AREA3D_Active_Reconstruction_Agent_with_Unified_Feed-Forward_3D_Perception_and_CVPR_2026_paper.html>
- Conformal point-cloud confidence: <https://doi.org/10.1016/j.patcog.2026.114249>
