from .geometry_head import CompactTTTState, SlotConditionedGeometryHead, StreamingGeometryHead, TrackAnchoredDepthHead
from .revisit_meta import RevisitMetaLearner, SignedResidualTransport
from .learned_update import LocalTrackUpdateRule
from .context_key import GeometryContextKey

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
