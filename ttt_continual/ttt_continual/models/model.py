"""The streaming reconstructor.

A stream arrives as chunks of posed frames. Each chunk is tokenised, run through
the trunk, and used to write the fast weight; query cameras -- which may be any
pose at any timestamp -- are decoded into Gaussians and rasterised. Nothing is
kept between chunks except the fast weight and, when enabled, the skill bank.

The order inside a chunk matters and is not arbitrary:

    describe the regime -> read a skill -> apply as a warm start
        -> run the trunk, which writes the fast weight
        -> decode and render
        -> consolidate what the chunk learned back into the bank

The read comes before the update because a retrieved skill is a starting point
the chunk then refines, not a substitute for adapting. The write comes after,
because what is worth storing is what the adaptation actually produced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .bank import SkillBank
from .blocks import Block
from .gaussians import GaussianHead, render, sinusoidal_time
from .regime import RegimeEncoder
from .ttt_layer import FastWeightState, TTTLayer


@dataclass
class ModelConfig:
    dim: int = 384
    depth: int = 12
    head_dim: int = 64
    mlp_expansion: float = 4.0
    patch_size: int = 8
    in_channels: int = 9              # plucker(6) + rgb(3)
    image_size: tuple = (224, 400)
    ttt_layers: tuple = ()            # which blocks carry a fast weight; empty -> all
    ttt_expansion: int = 4
    muon_steps: int = 5
    base_lr: float = 0.01
    sh_degree: int = 1
    max_depth: float = 10.0
    time_dim: int = 32
    grad_checkpoint: bool = False
    # skill memory
    use_bank: bool = True
    bank_layers: tuple = (1, 2, 7)    # configuration, not a constant: the layers
                                      # were localised on a different model
    n_slots: int = 16
    rank: int = 8
    write_beta: float = 0.05
    revive_after: int = 200


@dataclass
class StreamState:
    """Everything carried from one chunk to the next."""
    fast: Dict[int, FastWeightState] = field(default_factory=dict)
    n_chunks: int = 0
    last_regime: Optional[torch.Tensor] = None

    def detach(self) -> "StreamState":
        return StreamState({i: s.detach() for i, s in self.fast.items()},
                           self.n_chunks, self.last_regime)


class StreamingReconstructor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        self.tokens_per_view = (cfg.image_size[0] // p) * (cfg.image_size[1] // p)

        self.patch = nn.Linear(cfg.in_channels * p * p, cfg.dim, bias=False)
        self.time_proj = nn.Linear(cfg.time_dim, cfg.dim, bias=False)
        self.in_norm = nn.LayerNorm(cfg.dim)

        ttt_at = set(cfg.ttt_layers) if cfg.ttt_layers else set(range(cfg.depth))
        self.ttt_at = sorted(ttt_at)
        self.blocks = nn.ModuleList([
            Block(cfg.dim, cfg.head_dim, cfg.mlp_expansion,
                  ttt_layer=TTTLayer(cfg.dim, cfg.ttt_expansion, cfg.muon_steps,
                                     cfg.base_lr) if i in ttt_at else None)
            for i in range(cfg.depth)
        ])
        self.head = GaussianHead(cfg.dim, p, cfg.sh_degree, cfg.max_depth)

        self.regime = RegimeEncoder()
        self.bank = None
        if cfg.use_bank:
            shapes = {}
            for i in cfg.bank_layers:
                assert i in ttt_at, f"bank layer {i} has no fast weight"
                layer = self.blocks[i].ttt
                shapes[i] = {"w0": (layer.dim, layer.hidden),
                             "w1": (layer.hidden, layer.dim),
                             "w2": (layer.dim, layer.hidden)}
            self.bank = SkillBank(shapes, dim_key=self.regime.dim, n_slots=cfg.n_slots,
                                  rank=cfg.rank, write_beta=cfg.write_beta,
                                  revive_after=cfg.revive_after)

    # ---- tokenisation --------------------------------------------------
    def tokenise(self, feats: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """(b, v, c, h, w) -> (b, v * tokens_per_view, dim), with time added."""
        b, v, c, h, w = feats.shape
        if v == 0:
            # a chunk may legitimately carry no context: candidate evaluation
            # drops it, since with the fast weight only read there is no path
            # from a context token to a query token
            return feats.new_zeros(b, 0, self.cfg.dim)
        p = self.cfg.patch_size
        x = feats.reshape(b * v, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b * v, (h // p) * (w // p), c * p * p)
        x = self.patch(x)
        t = self.time_proj(sinusoidal_time(times.reshape(b * v), self.cfg.time_dim))
        x = x + t.unsqueeze(1)
        return self.in_norm(x.reshape(b, v * x.shape[1], self.cfg.dim))

    def init_stream(self, batch: int, device=None) -> StreamState:
        return StreamState({i: self.blocks[i].ttt.init_state(batch, device=device)
                            for i in self.ttt_at})

    # ---- one chunk -----------------------------------------------------
    def forward_chunk(self, chunk: Dict[str, torch.Tensor], state: StreamState,
                      regime: Optional[torch.Tensor] = None,
                      update: bool = True, use_bank: bool = True,
                      write: bool = False,
                      correction: Optional[Dict] = None) -> Dict:
        """Run one chunk of context frames and decode the requested queries.

        `chunk` holds `context` (features and times of the frames to adapt on),
        `query` (features and times of the cameras to decode at) and the rays
        for the query pixels.
        """
        ctx = self.tokenise(chunk["context_feats"], chunk["context_times"])
        qry = self.tokenise(chunk["query_feats"], chunk["query_times"])
        n_ctx = ctx.shape[1]
        x = torch.cat([ctx, qry], dim=1)

        if regime is not None and self.training:
            # The encoder's running scale is only meaningful if something feeds
            # it. Without this the descriptor is passed through unnormalised,
            # its components differ by two orders of magnitude, and every query
            # lands nearest the same slot regardless of the regime it describes.
            self.regime.observe(regime.detach())

        slot = None
        if correction is not None:
            pass                       # caller supplied one; used as given
        elif self.bank is not None and use_bank and regime is not None:
            key = self.regime(regime)
            slot, correction = self.bank.correction(key)

        before = {i: state.fast[i].detach() for i in self.ttt_at} if write else None

        checkpointing = self.cfg.grad_checkpoint and self.training and x.requires_grad
        for i, block in enumerate(self.blocks):
            corr = correction.get(i) if (correction is not None and i in correction) else None
            st = state.fast.get(i)
            if checkpointing:
                x, st = torch.utils.checkpoint.checkpoint(
                    block, x, self.tokens_per_view, st, update, corr,
                    use_reentrant=False)
            else:
                x, st = block(x, self.tokens_per_view, state=st, update=update,
                              correction=corr)
            if st is not None:
                state.fast[i] = st

        gaussians = self.head(x[:, n_ctx:], chunk["query_ray_o"], chunk["query_ray_d"])

        if write and self.bank is not None and regime is not None:
            delta = {i: (state.fast[i] - before[i]) for i in self.bank.layers}
            self.bank.write(self.regime(regime).detach(), delta)

        state.n_chunks += 1
        state.last_regime = regime
        return {"gaussians": gaussians, "state": state, "slot": slot,
                "correction": correction, "tokens": x}

    def render(self, gaussians, c2w, intr, height, width, **kw):
        return render(gaussians, c2w, intr, height, width,
                      sh_degree=self.cfg.sh_degree, **kw)

    # ---- introspection used by the measurement harness -------------------
    def snapshot(self, state: StreamState) -> Dict[int, Dict[str, torch.Tensor]]:
        return {i: {k: v.detach().clone() for k, v in state.fast[i].as_dict().items()}
                for i in self.ttt_at}

    def load_snapshot(self, state: StreamState, snap: Dict) -> StreamState:
        for i, mats in snap.items():
            state.fast[i] = FastWeightState(mats["w0"], mats["w1"], mats["w2"])
        return state

    def n_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        bank = sum(p.numel() for p in self.bank.parameters()) if self.bank else 0
        fast = sum(p.numel() for i in self.ttt_at
                   for p in (self.blocks[i].ttt.w0_init, self.blocks[i].ttt.w1_init,
                             self.blocks[i].ttt.w2_init))
        return {"total": total, "bank": bank, "fast_weight_init": fast,
                "trunk": total - bank}
