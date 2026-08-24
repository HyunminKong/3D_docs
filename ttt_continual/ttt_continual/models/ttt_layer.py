"""Test-time-training layer: a fast weight updated by a delta rule.

The layer holds a small SwiGLU network whose weights are rewritten as the stream
arrives:

    f(x) = (silu(x @ w0) * (x @ w2)) @ w1

Reading it is an ordinary forward pass with the queries. Writing it is one step
of gradient descent on ||f(k) - v||^2 with per-token learning rates the model
predicts, applied over a chunk at a time. The gradients are written out by hand
rather than taken from autograd so the whole chunk is a few matmuls, and so the
graph stays shallow enough to differentiate through during meta-training.

Two details are load-bearing and are kept from the reference implementation.

Newton-Schulz orthogonalisation (Muon) is applied to each gradient before it is
added. Without it the update is dominated by a few directions and the fast
weight drifts along them; with it the step is closer to an equal-magnitude
rotation of the weight, which is what makes long streams stable.

Column norms are restored after every update. The fast weight is allowed to
change direction freely but not to grow, which stops the accumulated updates
from quietly rescaling the layer's output over thousands of chunks.

This layer is also the only path by which one view influences another -- the
surrounding attention is intra-view by construction. That is deliberate: it
makes the fast weight the sole carrier of history, so anything measured about
forgetting is about this tensor and nothing else.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d/dx of silu, times an incoming gradient."""
    sigma = torch.sigmoid(x)
    return dy * sigma * (1.0 + x * (1.0 - sigma))


def orthogonalise(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz quintic iteration, an inexpensive stand-in for the
    orthogonal factor of the SVD.

    Returns something close to U V^T for G = U S V^T. `steps = 0` disables it.
    """
    if steps <= 0:
        return g
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-1, -2)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.transpose(-1, -2)
        x = a * x + (b * aa + c * aa @ aa) @ x
    if transposed:
        x = x.transpose(-1, -2)
    return x.to(g.dtype)


class FastWeightState:
    """The three matrices, carried along a stream.

    Held outside the module so that one module can serve many concurrent
    episodes, and so a state can be snapshotted, restored or corrected without
    touching the module that produced it.
    """

    __slots__ = ("w0", "w1", "w2")

    def __init__(self, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
        self.w0, self.w1, self.w2 = w0, w1, w2

    def clone(self) -> "FastWeightState":
        return FastWeightState(self.w0.clone(), self.w1.clone(), self.w2.clone())

    def detach(self) -> "FastWeightState":
        return FastWeightState(self.w0.detach(), self.w1.detach(), self.w2.detach())

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"w0": self.w0, "w1": self.w1, "w2": self.w2}

    def __sub__(self, other: "FastWeightState") -> Dict[str, torch.Tensor]:
        return {"w0": self.w0 - other.w0,
                "w1": self.w1 - other.w1,
                "w2": self.w2 - other.w2}


class TTTLayer(nn.Module):
    """Chunked fast-weight layer.

    Args:
        dim: model width.
        expansion: hidden width of the fast network, as a multiple of `dim`.
        muon_steps: Newton-Schulz iterations on each gradient; 0 turns it off.
        base_lr: starting scale of the predicted inner learning rate. Trainable,
            because how fast the fast weight should move is exactly the kind of
            thing the outer loop ought to decide.
    """

    def __init__(self, dim: int, expansion: int = 4, muon_steps: int = 5,
                 base_lr: float = 0.01, learn_lr: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden = int(dim * expansion)

        gain = math.sqrt(2.0)
        self.w0_init = nn.Parameter(torch.randn(dim, self.hidden) * gain / math.sqrt(dim))
        self.w1_init = nn.Parameter(torch.randn(self.hidden, dim) * gain / math.sqrt(self.hidden))
        self.w2_init = nn.Parameter(torch.randn(dim, self.hidden) * gain / math.sqrt(dim))

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.RMSNorm(dim)

        # one learning rate per matrix per token
        self.lr_head = nn.Linear(dim, 3)
        nn.init.zeros_(self.lr_head.weight)
        inv = base_lr + math.log(-math.expm1(-base_lr)) if base_lr < 20 else base_lr
        nn.init.constant_(self.lr_head.bias, inv)
        self.lr_head.bias.requires_grad_(learn_lr)
        self.lr_head.weight.requires_grad_(learn_lr)

        self.muon_steps = muon_steps

    def init_state(self, batch: int, device=None, dtype=None) -> FastWeightState:
        expand = lambda p: p.unsqueeze(0).expand(batch, -1, -1).to(
            device=device or p.device, dtype=dtype or p.dtype).contiguous()
        return FastWeightState(expand(self.w0_init), expand(self.w1_init), expand(self.w2_init))

    def _project(self, x: torch.Tensor):
        qkv = F.silu(self.to_qkv(x))
        q, k, v = qkv.chunk(3, dim=-1)
        # q and k are unit-normalised so the update magnitude is set by the
        # learning rate head alone, not by how large the activations happen to be
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-5)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
        lr = F.softplus(self.lr_head(x.float()))
        return q, k, v, lr

    def update(self, state: FastWeightState, x: torch.Tensor) -> FastWeightState:
        """One delta-rule step over a chunk. Does not produce an output."""
        _, k, v, lr = self._project(x)
        return self._apply_update(state, k, v, lr)

    def _apply_update(self, state, k, v, lr) -> FastWeightState:
        w0, w1, w2 = state.w0, state.w1, state.w2
        # preserved through the update, so the layer can rotate but not inflate
        n0 = w0.detach().norm(dim=1, keepdim=True)
        n1 = w1.detach().norm(dim=1, keepdim=True)
        n2 = w2.detach().norm(dim=1, keepdim=True)

        k = k.to(w0.dtype)
        v = v.to(w0.dtype)
        lr0, lr1, lr2 = lr[..., 0:1], lr[..., 1:2], lr[..., 2:3]

        gate_pre = k @ w0
        value_pre = k @ w2
        hidden = F.silu(gate_pre) * value_pre

        # gradient of ||f(k) - v||^2 wrt each matrix, written out
        d_hidden = v @ w1.transpose(-1, -2)
        d_value = d_hidden * F.silu(gate_pre)
        d_gate = silu_backprop(d_hidden * value_pre, gate_pre)

        g1 = (hidden * lr1.to(hidden.dtype)).transpose(-1, -2) @ v
        g0 = (k * lr0.to(k.dtype)).transpose(-1, -2) @ d_gate
        g2 = (k * lr2.to(k.dtype)).transpose(-1, -2) @ d_value

        w0 = w0 + orthogonalise(g0, self.muon_steps)
        w1 = w1 + orthogonalise(g1, self.muon_steps)
        w2 = w2 + orthogonalise(g2, self.muon_steps)

        w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * n0
        w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * n1
        w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * n2
        return FastWeightState(w0, w1, w2)

    def read(self, state: FastWeightState, x: torch.Tensor) -> torch.Tensor:
        """Query the fast weight without changing it."""
        q, _, _, _ = self._project(x)
        q = q.to(state.w0.dtype)
        out = (F.silu(q @ state.w0) * (q @ state.w2)) @ state.w1
        return self.proj(self.out_norm(out.float()).to(x.dtype))

    def forward(self, x: torch.Tensor, state: FastWeightState,
                update: bool = True, correction: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, FastWeightState]:
        """Read, then optionally write.

        `correction` is the skill-bank term. It is added *before* the update, so
        a retrieved skill is a warm start that the chunk's own adaptation then
        refines -- not a replacement for it. Substituting a remembered state
        outright was measured and fails; adding a low-rank step toward one does
        not.
        """
        if correction is not None:
            state = FastWeightState(
                state.w0 + correction["w0"].to(state.w0.dtype),
                state.w1 + correction["w1"].to(state.w1.dtype),
                state.w2 + correction["w2"].to(state.w2.dtype),
            )
        out = self.read(state, x)
        if update:
            state = self.update(state, x)
        return out, state
