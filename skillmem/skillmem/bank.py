"""The skill bank: a fixed number of slots, read one at a time.

Three things about it are forced by measurement rather than preference.

It stores a *correction*, not a state. Interpolating toward a remembered fast
weight collapses between the two — the midpoint sits 3.4 dB below the worse
endpoint — because two adapted states occupy different basins. The difference
between them is a small vector that can be added near the current state, and
that intervention behaves.

It stores three layers out of twenty-four. The second half of the trunk carries
no scene-specific signal at all (about 0% retained, 10 of 10 scenes), while
layers 1, 2 and 7 together carry 78.8%. That is what turns 566 MB into 18 MB and
makes the parameter-matched comparison winnable. The set is configuration, not a
constant: it was localised on a model this work is about to change.

It always offers the option of doing nothing. A correction retrieved from the
wrong place costs 1.04 dB where a random direction of the same size costs 0.06,
so a bad retrieval is actively harmful and abstaining has to be reachable. The
null skill is slot zero, its value fixed at zero, competing on equal terms.

Reading is top-1 with a straight-through estimator rather than a softmax
mixture, so that "the retrieved skill" stays a single slot and the contrastive
objective has something definite to be stated over. Top-1 sharpens the usual
codebook failure, where a few slots absorb every write and the rest keep their
initial values, so usage is tracked and unused slots are revived.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

W_KEYS = ("w0", "w1", "w2")


class SkillBank(nn.Module):
    """K skills plus a null, over a chosen subset of TTT layers.

    Values are low-rank factors trained by gradient through the read path. The
    EMA write is consolidation on top of that: repeated visits to one regime fold
    into the same slot, so a slot ends up holding what that regime's adaptations
    have in common rather than any single episode. Set `write_beta` to zero to
    disable it and leave the bank purely gradient-trained.
    """

    def __init__(self, layer_shapes, dim_key, n_slots=16, rank=8,
                 write_beta=0.05, revive_after=200, gate_margin=-4.0):
        # gate_margin sits below the negative distance a plausible match scores,
        # so slots can win at initialisation and receive gradient. Starting it at
        # zero makes the null option unbeatable -- every score is -distance, hence
        # negative -- and nothing but the null is ever read or trained. The value
        # is learned from here, so abstaining is something the objective has to
        # decide is worth it rather than the state it begins in.
        super().__init__()
        self.layers = sorted(layer_shapes)
        self.shapes = layer_shapes
        self.n_slots = n_slots
        self.rank = rank
        self.write_beta = write_beta
        self.revive_after = revive_after

        self.keys = nn.Parameter(torch.randn(n_slots, dim_key) * 0.1)
        self.null_logit = nn.Parameter(torch.tensor(float(gate_margin)))

        # one pair of factors per (slot, layer, matrix)
        self.factors = nn.ParameterDict()
        for layer in self.layers:
            for name, (d_in, d_out) in layer_shapes[layer].items():
                self.factors[f"{layer}_{name}_u"] = nn.Parameter(
                    torch.randn(n_slots, d_in, rank) * (d_in ** -0.5))
                # Small random rather than the LoRA convention of exactly zero.
                # At zero every slot produces the identical (null) correction, so
                # the choice of slot cannot change the output and no gradient
                # reaches the keys, the gate or the scale -- retrieval would never
                # start learning. The correction is kept negligible by log_scale
                # instead, which begins at exp(-3).
                self.factors[f"{layer}_{name}_v"] = nn.Parameter(
                    torch.randn(n_slots, rank, d_out) * (rank ** -0.5) * 0.02)
        # per-layer strength, learned; starts small because the measured usable
        # correction was found at lambda ~ 0.05
        self.log_scale = nn.Parameter(torch.full((len(self.layers),), -3.0))

        self.register_buffer("usage", torch.zeros(n_slots))
        self.register_buffer("age", torch.zeros(n_slots))

    # ---- retrieval ----------------------------------------------------
    def logits(self, query):
        """Similarity to every slot, with the null option prepended."""
        d = torch.cdist(query, self.keys)                     # (batch, K)
        null = self.null_logit.expand(query.shape[0], 1)
        return torch.cat([null, -d], dim=1)                   # index 0 == abstain

    def read(self, query, hard=True):
        """Pick one skill. Returns (index, one-hot with a gradient path).

        The one-hot is straight-through: the forward value is the hard choice, so
        exactly one slot's correction is applied, while the backward pass sees
        the softmax and can move the keys.
        """
        logit = self.logits(query)
        soft = F.softmax(logit, dim=-1)
        if not hard:
            return soft.argmax(-1), soft
        index = logit.argmax(-1)
        onehot = F.one_hot(index, logit.shape[-1]).to(soft.dtype)
        return index, onehot + soft - soft.detach()

    def correction(self, query, hard=True):
        """Per-layer delta to add on top of the current fast weight.

        Slot zero is the null skill and contributes nothing, so an abstaining
        query leaves the fast weight exactly as the plain TTT update left it.
        """
        index, weights = self.read(query, hard=hard)
        slot_w = weights[:, 1:]                               # drop the null column
        out = {}
        for pos, layer in enumerate(self.layers):
            scale = self.log_scale[pos].exp()
            entry = {}
            for name in self.shapes[layer]:
                u = self.factors[f"{layer}_{name}_u"]         # (K, d_in, r)
                v = self.factors[f"{layer}_{name}_v"]         # (K, r, d_out)
                uu = torch.einsum("bk,kir->bir", slot_w, u)
                vv = torch.einsum("bk,kro->bro", slot_w, v)
                entry[name] = scale * torch.bmm(uu, vv)
            out[layer] = entry
        return index, out

    # ---- consolidation -------------------------------------------------
    @torch.no_grad()
    def write(self, query, delta, index=None):
        """Fold an observed adaptation into its nearest slot.

        `delta` maps layer -> {matrix: (d_in, d_out)}, the change the plain TTT
        update produced for this chunk. Only the stored layers are read from it.
        Training-time only: deciding whether a write helped needs the
        counterfactual of not having written it, which does not exist at
        inference, so the bank is frozen once built.
        """
        if self.write_beta <= 0:
            return None
        if index is None:
            index = torch.cdist(query, self.keys).argmin(-1)
        for b, slot in enumerate(index.tolist()):
            for layer in self.layers:
                if layer not in delta:
                    continue
                for name in self.shapes[layer]:
                    target = delta[layer][name]
                    target = target[b] if target.dim() == 3 else target
                    u, s, vh = torch.linalg.svd(target.float(), full_matrices=False)
                    r = min(self.rank, s.shape[-1])
                    root = s[:r].sqrt()
                    new_u = u[:, :r] * root
                    new_v = root.unsqueeze(-1) * vh[:r]
                    pu = self.factors[f"{layer}_{name}_u"]
                    pv = self.factors[f"{layer}_{name}_v"]
                    pu[slot].mul_(1 - self.write_beta).add_(self.write_beta * new_u.to(pu.dtype))
                    pv[slot].mul_(1 - self.write_beta).add_(self.write_beta * new_v.to(pv.dtype))
            self.keys[slot].mul_(1 - self.write_beta).add_(self.write_beta * query[b])
            self.usage[slot] += 1
        self.age += 1
        self.age[index] = 0
        return index

    @torch.no_grad()
    def revive(self, query):
        """Reseed slots nothing has claimed.

        Top-1 reading rewards whichever slots win early and starves the rest;
        left alone the bank quietly shrinks to a handful of live entries. A slot
        that has gone `revive_after` writes untouched is moved onto a query the
        current bank fits worst, which is where a new skill would be useful.
        """
        dead = (self.age > self.revive_after).nonzero().flatten()
        if not len(dead) or not len(query):
            return 0
        far = torch.cdist(query, self.keys).min(dim=1).values.argsort(descending=True)
        for n, slot in enumerate(dead.tolist()):
            src = query[far[n % len(far)]]
            self.keys[slot] = src + torch.randn_like(src) * 0.01
            for layer in self.layers:
                for name in self.shapes[layer]:
                    self.factors[f"{layer}_{name}_v"][slot].zero_()   # revive as a no-op
            self.age[slot] = 0
            self.usage[slot] = 0
        return len(dead)

    def occupancy(self):
        """Fraction of slots that have ever been written — collapse shows up here."""
        return float((self.usage > 0).float().mean())
