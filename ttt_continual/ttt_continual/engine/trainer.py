"""The training loop.

One step is one episode: a stream is opened, chunks are fed through in order,
and every chunk's queries are rendered and scored. Two things about this differ
from an ordinary image-model loop.

The fast weight is a recurrence, so the graph would otherwise span the whole
episode. It is detached every `truncate` chunks, which bounds memory and matches
what the reusability objective actually needs -- credit has to travel from a
write to a later read, and that path runs through the bank, which is a
persistent parameter, not through the intervening chunks.

The contrastive term needs the loss each *candidate* skill would have produced,
which means re-rendering the same queries with different corrections. Rendering
all of them would cost K+1 passes per chunk, so a subset is sampled: the skill
that was retrieved, the null, and a few others. The comparison stays honest
because the null is always present -- what a skill is measured against is the
option of not using one.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..data.nuscenes import build_features
from ..losses.reconstruction import LossWeights, reconstruction_loss
from ..losses.reusability import contrastive_reuse, foreign_penalty
from ..metrics.image import MetricTracker, psnr


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    total_steps: int = 40000
    grad_accum: int = 4
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    truncate: int = 2                 # chunks per differentiated segment
    reuse_weight: float = 0.1
    reuse_candidates: int = 3         # sampled alternatives, plus the null
    reuse_start_step: int = 2000      # let reconstruction settle first
    write_from_step: int = 1000
    eval_every: int = 1000
    save_every: int = 2000
    log_every: int = 20
    loss_weights: LossWeights = field(default_factory=LossWeights)


class Trainer:
    def __init__(self, model, optimizer, scheduler, cfg: TrainConfig,
                 logger=None, device: str = "cuda"):
        self.model = model
        self.opt = optimizer
        self.sched = scheduler
        self.cfg = cfg
        self.logger = logger
        self.device = device
        self.step = 0
        self.amp = getattr(torch, cfg.amp_dtype)
        # regime keys observed since the last optimiser step; revival reseeds
        # unused slots onto the queries the bank currently fits worst
        self._seen_keys = []

    # ---- rendering one chunk -------------------------------------------
    def _render_and_score(self, out, chunk, weights) -> Dict:
        g = out["gaussians"]
        render = self.model.render(
            g, chunk["query_c2w"][0], chunk["query_intr"][0],
            chunk["query_rgb"].shape[-2], chunk["query_rgb"].shape[-1])
        terms = reconstruction_loss(render, chunk["query_rgb"][0], g, weights)
        return render, terms

    def _chunk_inputs(self, chunk) -> Dict:
        ctx_feats, _, _ = build_features(chunk["context_rgb"], chunk["context_c2w"],
                                         chunk["context_intr"])
        q_feats, q_o, q_d = build_features(chunk["query_rgb"], chunk["query_c2w"],
                                           chunk["query_intr"])
        b, v, _, h, w = chunk["query_rgb"].shape
        # one ray per emitted Gaussian, unpacked in the head's order:
        # (view, patch_row, patch_col, pixel_row, pixel_col)
        p = self.model.cfg.patch_size
        ray = lambda r: (r.reshape(b, v, 3, h // p, p, w // p, p)
                          .permute(0, 1, 3, 5, 4, 6, 2).reshape(b, -1, 3))
        return {
            "context_feats": ctx_feats, "context_times": chunk["context_times"],
            "query_feats": q_feats, "query_times": chunk["query_times"],
            "query_ray_o": ray(q_o), "query_ray_d": ray(q_d),
        }

    # ---- candidate evaluation for the reusability term -------------------
    def _candidate_losses(self, chunk, inputs, state, key, weights) -> Optional[Dict]:
        bank = self.model.bank
        if bank is None:
            return None
        n = bank.n_slots + 1
        chosen = int(bank.logits(key).argmax(-1))
        pool = [i for i in range(n) if i not in (0, chosen)]
        cand = [0, chosen] + random.sample(pool, min(self.cfg.reuse_candidates, len(pool)))

        empty = inputs["query_feats"][:, :0]
        query_only = dict(inputs)
        query_only["context_feats"] = empty
        query_only["context_times"] = inputs["query_times"][:, :0]

        losses = []
        for slot in cand:
            onehot = torch.zeros(1, n, device=key.device)
            onehot[0, slot] = 1.0
            corr = self._correction_from_onehot(onehot)
            # Context frames are dropped here. With `update=False` there is no
            # path from a context token to a query token -- attention is
            # intra-view, the MLP is pointwise, and the fast weight is only read
            # -- so they cannot change the render, and carrying them would
            # multiply the cost of every candidate by the context size.
            out = self.model.forward_chunk(
                query_only, state.detach(), regime=None, update=False,
                use_bank=False, correction=corr)
            _, terms = self._render_and_score(out, chunk, weights)
            losses.append(terms["total"])
        stacked = torch.stack(losses).unsqueeze(0)
        order = {s: i for i, s in enumerate(cand)}
        return {"losses": stacked, "chosen_col": torch.tensor([order[chosen]],
                                                              device=key.device)}

    def _correction_from_onehot(self, onehot) -> Dict:
        bank = self.model.bank
        slot_w = onehot[:, 1:]
        out = {}
        for pos, layer in enumerate(bank.layers):
            scale = bank.log_scale[pos].exp()
            entry = {}
            for name in bank.shapes[layer]:
                u = bank.factors[f"{layer}_{name}_u"]
                v = bank.factors[f"{layer}_{name}_v"]
                uu = torch.einsum("bk,kir->bir", slot_w, u)
                vv = torch.einsum("bk,kro->bro", slot_w, v)
                entry[name] = scale * torch.bmm(uu, vv)
            out[layer] = entry
        return out

    # ---- one episode ----------------------------------------------------
    def train_episode(self, episode) -> Dict[str, float]:
        cfg = self.cfg
        chunks = episode["chunks"]
        state = self.model.init_stream(1, device=self.device)
        tracker = MetricTracker()
        total = torch.zeros((), device=self.device)
        n_scored = 0

        for i, chunk in enumerate(chunks):
            chunk = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in chunk.items()}
            inputs = self._chunk_inputs(chunk)
            regime = chunk["regime"]
            with torch.no_grad():
                self._seen_keys.append(self.model.regime(regime).detach())

            with torch.autocast("cuda", dtype=self.amp):
                out = self.model.forward_chunk(
                    inputs, state, regime=regime, update=True,
                    use_bank=self.model.bank is not None,
                    write=(self.step >= cfg.write_from_step))
                state = out["state"]
                render, terms = self._render_and_score(out, chunk, cfg.loss_weights)

            total = total + terms["total"]
            n_scored += 1

            with torch.no_grad():
                per_view = psnr(render["rgb"], chunk["query_rgb"][0])
                is_rev = chunk["query_is_revisit"][0].bool()
                if (~is_rev).any():
                    tracker.update("psnr_current", per_view[~is_rev].mean())
                if is_rev.any():
                    tracker.update("psnr_revisit", per_view[is_rev].mean())
                for k, v in terms.items():
                    tracker.update(f"loss_{k}", float(v))

            if cfg.reuse_weight > 0 and self.step >= cfg.reuse_start_step:
                with torch.autocast("cuda", dtype=self.amp):
                    key = self.model.regime(regime)
                    cand = self._candidate_losses(chunk, inputs, state, key,
                                                  cfg.loss_weights)
                if cand is not None:
                    spread = float(cand["losses"].max() - cand["losses"].min())
                    if spread < 1e-8 and self.step == self.cfg.reuse_start_step:
                        print("warning: candidate skills give identical loss; the "
                              "reusability term carries no signal. Check that the "
                              "Gaussian head is not input-independent.", flush=True)
                    reuse = contrastive_reuse(cand["losses"], cand["chosen_col"])
                    total = total + cfg.reuse_weight * reuse["loss"]
                    for k in ("margin", "retrieval_accuracy", "beats_null"):
                        tracker.update(f"reuse_{k}", float(reuse[k]))

            # bound the graph; the bank, not the recurrence, carries credit
            if (i + 1) % cfg.truncate == 0:
                state = state.detach()

        loss = total / max(n_scored, 1)
        (loss / cfg.grad_accum).backward()
        tracker.update("loss", float(loss))
        return tracker.as_dict()

    def optimiser_step(self) -> float:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g["params"]], self.cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        if self.model.bank is not None:
            with torch.no_grad():
                self.model.bank.flush()          # staged EMA writes, graph now gone
                if self._seen_keys:
                    self.model.bank.revive(torch.cat(self._seen_keys))
                self._seen_keys = []
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self, loader, max_episodes: int = 20) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        for n, episode in enumerate(loader):
            if n >= max_episodes:
                break
            state = self.model.init_stream(1, device=self.device)
            for chunk in episode["chunks"]:
                chunk = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                         for k, v in chunk.items()}
                inputs = self._chunk_inputs(chunk)
                with torch.autocast("cuda", dtype=self.amp):
                    out = self.model.forward_chunk(
                        inputs, state, regime=chunk["regime"], update=True,
                        use_bank=self.model.bank is not None, write=False)
                    state = out["state"]
                    render, terms = self._render_and_score(out, chunk,
                                                           self.cfg.loss_weights)
                per_view = psnr(render["rgb"], chunk["query_rgb"][0])
                is_rev = chunk["query_is_revisit"][0].bool()
                if (~is_rev).any():
                    tracker.update("psnr_current", per_view[~is_rev].mean())
                if is_rev.any():
                    tracker.update("psnr_revisit", per_view[is_rev].mean())
                tracker.update("loss", float(terms["total"]))
        if self.model.bank is not None:
            tracker.update("bank_occupancy", self.model.bank.occupancy())
        self.model.train()
        return tracker.as_dict()
