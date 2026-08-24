"""A compact, explicitly reusable TTT state for a frozen geometry backbone.

The state is not a hidden copy of all model weights.  It is a small vector
which FiLM-modulates only the new geometry head.  That design makes the object
which is adapted at test time the same object that can later be evaluated for
reusability or stored in a memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CompactTTTState:
    value: Tensor

    @classmethod
    def zeros(cls, batch_size: int, dimension: int, *, device=None, dtype=None) -> "CompactTTTState":
        return cls(torch.zeros(batch_size, dimension, device=device, dtype=dtype))

    def detach(self) -> "CompactTTTState":
        return CompactTTTState(self.value.detach())


class StreamingGeometryHead(nn.Module):
    """Decode frozen per-token features into point, depth, pose and confidence.

    Input tokens may be supplied by VGGT, DINOv3, or another frozen foundation
    model.  The shape is ``[batch, views, tokens, feature_dim]``.  Only this
    head and its compact state are trainable/adaptable.
    """

    def __init__(self, feature_dim: int, state_dim: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.input_norm = nn.LayerNorm(feature_dim)
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, 2 * feature_dim)
        )
        self.token_trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.point_head = nn.Linear(hidden_dim, 3)
        self.depth_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.pose_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> CompactTTTState:
        return CompactTTTState.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def forward(self, features: Tensor, state: CompactTTTState) -> dict[str, Tensor]:
        if features.ndim != 4:
            raise ValueError("features must be [batch, views, tokens, feature_dim]")
        if state.value.shape != (features.shape[0], self.state_dim):
            raise ValueError("state batch/dimension does not match head")
        scale, shift = self.to_scale_shift(state.value).chunk(2, dim=-1)
        x = self.input_norm(features) * (1 + scale[:, None, None]) + shift[:, None, None]
        token = self.token_trunk(x)
        pooled = token.mean(dim=2)
        return {
            "pointmap": self.point_head(token),
            "depth": F.softplus(self.depth_head(token)) + 1e-4,
            "confidence": torch.sigmoid(self.confidence_head(token)),
            "relative_pose": self.pose_head(pooled),
        }

    def adapt(
        self,
        features: Tensor,
        state: CompactTTTState,
        objective: Callable[[dict[str, Tensor]], Tensor],
        *,
        steps: int = 1,
        learning_rate: float = 1e-2,
        create_graph: bool = False,
        retain_state_gradient: bool = False,
    ) -> tuple[CompactTTTState, list[Tensor]]:
        """Update only ``z``; no backbone or head parameter is mutated.

        ``objective`` will later be a reprojection/depth/track consistency loss.
        Keeping it as an explicit callback prevents the benchmark from silently
        treating a supervised target as an online TTT signal.
        """
        # Evaluation TTT is first-order and releases its graph.  Meta-training
        # sets ``create_graph`` so a revisit loss can shape the update rule and
        # the coordinate system itself.
        z = state.value if (create_graph or retain_state_gradient) else state.value.detach()
        history: list[Tensor] = []
        for _ in range(steps):
            z = z.requires_grad_(True)
            loss = objective(self(features, CompactTTTState(z)))
            if loss.ndim != 0:
                raise ValueError("TTT objective must return a scalar")
            grad, = torch.autograd.grad(loss, z, only_inputs=True, create_graph=create_graph)
            # First-order meta-TTT intentionally drops the Hessian through an
            # online reprojection update while retaining gradients from a
            # transported initial state into the held-out outer loss.
            z = z - learning_rate * (grad if create_graph else grad.detach())
            if not create_graph and not retain_state_gradient:
                z = z.detach()
            history.append(loss.detach())
        return CompactTTTState(z), history


class SlotConditionedGeometryHead(StreamingGeometryHead):
    """Geometry decoder with a feature-routed, local compact TTT state.

    The global FiLM state above applies the same modulation to every token, so
    unrelated B observations can induce almost the same update as a revisit.
    Here the adaptable state is ``K`` small slots.  A frozen-feature-dependent
    router selects a mixture of slots per token, making the *effect* and the
    gradient of each TTT update spatial/content-local while keeping the state
    compact enough for later oracle/retrieval experiments.
    """

    def __init__(self, feature_dim: int, state_dim: int = 16, slots: int = 8,
                 routing_dim: int = 64, hidden_dim: int = 512, routing_temperature: float = 0.1):
        # Build common decoder heads but replace the global state projection.
        super().__init__(feature_dim, state_dim=state_dim, hidden_dim=hidden_dim)
        self.slots = slots
        self.routing_dim = routing_dim
        self.routing_temperature = routing_temperature
        self.to_scale_shift = nn.Linear(state_dim, 2 * feature_dim)
        self.token_router = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, routing_dim, bias=False))
        self.slot_keys = nn.Parameter(torch.randn(slots, routing_dim) * (routing_dim ** -0.5))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> CompactTTTState:
        return CompactTTTState(torch.zeros(batch_size, self.slots, self.state_dim, device=device, dtype=dtype))

    def forward(self, features: Tensor, state: CompactTTTState) -> dict[str, Tensor]:
        if features.ndim != 4:
            raise ValueError("features must be [batch, views, tokens, feature_dim]")
        expected = (features.shape[0], self.slots, self.state_dim)
        if state.value.shape != expected:
            raise ValueError(f"state must be {expected}, got {tuple(state.value.shape)}")
        token_keys = F.normalize(self.token_router(features), dim=-1)
        slot_keys = F.normalize(self.slot_keys, dim=-1)
        assignment = torch.softmax(torch.einsum("bvpr,kr->bvpk", token_keys, slot_keys) / self.routing_temperature, dim=-1)
        local_state = torch.einsum("bvpk,bkd->bvpd", assignment, state.value)
        scale, shift = self.to_scale_shift(local_state).chunk(2, dim=-1)
        x = self.input_norm(features) * (1 + scale) + shift
        token = self.token_trunk(x)
        pooled = token.mean(dim=2)
        return {
            "pointmap": self.point_head(token),
            "depth": F.softplus(self.depth_head(token)) + 1e-4,
            "confidence": torch.sigmoid(self.confidence_head(token)),
            "relative_pose": self.pose_head(pooled),
            "slot_assignment": assignment,
        }


class TrackAnchoredDepthHead(StreamingGeometryHead):
    """Direct local-depth residual state routed by frozen-feature slots."""

    def __init__(self, feature_dim: int, slots: int = 8, routing_dim: int = 64, hidden_dim: int = 512,
                 routing_temperature: float = 0.1):
        super().__init__(feature_dim, state_dim=slots, hidden_dim=hidden_dim)
        self.slots, self.routing_temperature = slots, routing_temperature
        self.token_router = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, routing_dim, bias=False))
        self.slot_keys = nn.Parameter(torch.randn(slots, routing_dim) * (routing_dim ** -0.5))

    def forward(self, features: Tensor, state: CompactTTTState) -> dict[str, Tensor]:
        if state.value.shape != (features.shape[0], self.slots):
            raise ValueError("anchored state must be [batch, slots]")
        token_keys = F.normalize(self.token_router(features), dim=-1)
        slot_keys = F.normalize(self.slot_keys, dim=-1)
        assignment = torch.softmax(torch.einsum("bvpr,kr->bvpk", token_keys, slot_keys) / self.routing_temperature, dim=-1)
        local_log_residual = torch.einsum("bvpk,bk->bvp", assignment, state.value).clamp(-2, 2)
        token = self.token_trunk(self.input_norm(features))
        base_depth = F.softplus(self.depth_head(token)) + 1e-4
        return {"pointmap": self.point_head(token), "depth": base_depth * local_log_residual.exp().unsqueeze(-1),
                "confidence": torch.sigmoid(self.confidence_head(token)), "relative_pose": self.pose_head(token.mean(dim=2)),
                "slot_assignment": assignment}
