from .geometry_head import CompactTTTState, SlotConditionedGeometryHead, StreamingGeometryHead, TrackAnchoredDepthHead
from .revisit_meta import RevisitMetaLearner, SignedResidualTransport
from .learned_update import LocalTrackUpdateRule
from .context_key import GeometryContextKey
from .plasticity_atom import (
    PlasticityAtom,
    SpatialPlasticityHead,
    build_plasticity_atom,
    combine_atom_confidence,
    token_track_support,
)
from .geometry_transport import (
    FeatureMatches,
    Sim3Alignment,
    TransportResult,
    align_atoms,
    apply_sim3,
    backproject_tokens,
    geometry_transport,
    local_knn_scale,
    mutual_feature_matches,
    robust_sim3,
    visual_transport,
    weighted_sim3,
)
from .utility_router import (
    ObservableUtilityRiskRouter,
    RoutingDecision,
    UtilityRiskPrediction,
    apply_bounded_memory_residual,
)

def build_geometry_head(kind: str, feature_dim: int):
    if kind == "global":
        return StreamingGeometryHead(feature_dim, state_dim=32, hidden_dim=512)
    if kind == "slot":
        return SlotConditionedGeometryHead(feature_dim, state_dim=16, slots=8, hidden_dim=512)
    if kind == "anchored":
        return TrackAnchoredDepthHead(feature_dim, slots=8, hidden_dim=512)
    raise ValueError(f"unknown geometry head {kind!r}")


__all__ = ["CompactTTTState", "StreamingGeometryHead", "SlotConditionedGeometryHead", "TrackAnchoredDepthHead", "build_geometry_head",
           "RevisitMetaLearner", "SignedResidualTransport", "LocalTrackUpdateRule", "GeometryContextKey"]
__all__ += [
    "PlasticityAtom", "SpatialPlasticityHead", "build_plasticity_atom",
    "combine_atom_confidence", "token_track_support",
]
__all__ += [
    "FeatureMatches", "Sim3Alignment", "TransportResult", "align_atoms", "apply_sim3",
    "backproject_tokens", "geometry_transport", "local_knn_scale", "mutual_feature_matches",
    "robust_sim3", "visual_transport", "weighted_sim3",
]
__all__ += [
    "ObservableUtilityRiskRouter", "RoutingDecision", "UtilityRiskPrediction",
    "apply_bounded_memory_residual",
]
