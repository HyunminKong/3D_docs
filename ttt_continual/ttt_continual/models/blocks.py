"""Trunk blocks.

The attention here is deliberately restricted to a single view. Letting tokens
attend across frames would give history a second route into the present, and the
whole line of work depends on the fast weight being the only one -- with two
routes, a measurement of what was forgotten is a measurement of nothing in
particular. The cost is that all cross-view reasoning has to pass through a
fixed-size tensor, which is precisely the constraint under study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraViewAttention(nn.Module):
    """Self-attention within each view's own tokens."""

    def __init__(self, dim: int, head_dim: int = 64, qk_norm: bool = True):
        super().__init__()
        assert dim % head_dim == 0, "dim must divide evenly into heads"
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, tokens_per_view: int) -> torch.Tensor:
        b, n, d = x.shape
        assert n % tokens_per_view == 0, "token count must be a multiple of the view size"
        v = n // tokens_per_view
        x = x.reshape(b * v, tokens_per_view, d)

        qkv = self.to_qkv(x).reshape(b * v, tokens_per_view, 3, self.heads, self.head_dim)
        q, k, val = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, val)
        out = out.transpose(1, 2).reshape(b * v, tokens_per_view, d)
        return self.proj(out).reshape(b, n, d)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: intra-view attention, fast weight, MLP.

    The fast-weight sublayer is optional so that a trunk can mix plain blocks
    with adaptive ones. Where it is absent the block is a standard transformer
    block operating on one frame at a time.
    """

    def __init__(self, dim: int, head_dim: int = 64, mlp_expansion: float = 4.0,
                 ttt_layer: nn.Module = None):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = IntraViewAttention(dim, head_dim)
        self.ttt = ttt_layer
        self.norm_ttt = nn.LayerNorm(dim) if ttt_layer is not None else None
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, mlp_expansion)

    def forward(self, x, tokens_per_view, state=None, update=True, correction=None):
        x = x + self.attn(self.norm_attn(x), tokens_per_view)
        if self.ttt is not None:
            out, state = self.ttt(self.norm_ttt(x), state, update=update,
                                  correction=correction)
            x = x + out
        x = x + self.mlp(self.norm_mlp(x))
        return x, state
