"""Capture / inject per-layer TTT fast weights in tttLRM.

The AR schedule (`model.lact_ttt.ar_ttt_op`) updates the fast weight over the
input tokens in chunks and then performs a single apply over the whole
sequence.  That makes the readout separable from the update: once a fast weight
has been captured we can re-render with an arbitrary W without re-running the
chunked update.

Two facts about the architecture make this a clean instrument:

* `SelfAttention` is applied per image (`model/block.py`, `length_dim == "l"`)
  and `MLP` is pointwise, so the fast weight is the only path by which one view
  influences another.
* Gaussians are decoded from the query tokens only (`model/model.py`), so
  swapping W at apply time fully determines the rendered output.

The fast weight is layer-local, but `shape_info` is a single dict shared by all
24 blocks, so injection cannot go through it directly.  We wrap each
`FastWeightGluMLPMultihead` instance instead and set the keys around its own
call.
"""

import collections

import torch

from model.lact_ttt import FastWeightGluMLPMultihead, TTTOperator

W_KEYS = ("w0", "w1", "w2")


def apply_only_config(ttt_config):
    """Strip the update ops, keeping the final full-sequence apply."""
    length = max(op.end for op in ttt_config)
    return [TTTOperator(start=0, end=length, fast_weight=False, update=False, apply=True)]


class FastWeightController:
    """Attach to a tttLRM instance to capture and inject fast weights.

    Usage::

        ctl = FastWeightController(model)
        with ctl.capturing():
            model(batch)
        w_a = ctl.snapshot()

        with ctl.injecting(w_a, apply_only=True):
            model(batch)
    """

    def __init__(self, model):
        self.layers = [m for m in model.modules() if isinstance(m, FastWeightGluMLPMultihead)]
        if not self.layers:
            raise RuntimeError("no FastWeightGluMLPMultihead found; is this a ttt model?")
        self._captured = {}
        self._inject = None
        self._apply_only = False
        self._capture = False
        self._patch()

    # -- plumbing ---------------------------------------------------------
    def _patch(self):
        for idx, layer in enumerate(self.layers):
            layer._oracle_idx = idx
            original = layer.forward

            def wrapped(x, vis_dict=None, shape_info=None, *args, _layer=layer, _orig=original):
                i = _layer._oracle_idx
                restore = {}
                touched = []
                if self._inject is not None:
                    for key in W_KEYS:
                        if key in shape_info:
                            restore[key] = shape_info[key]
                        touched.append(key)
                        shape_info[key] = self._inject[i][key]
                if self._apply_only:
                    restore["ttt_config"] = shape_info["ttt_config"]
                    touched.append("ttt_config")
                    shape_info["ttt_config"] = apply_only_config(shape_info["ttt_config"])
                try:
                    out, vd = _orig(x, vis_dict, shape_info, *args)
                finally:
                    for key in touched:
                        if key in restore:
                            shape_info[key] = restore[key]
                        else:
                            shape_info.pop(key, None)
                if self._capture:
                    self._captured[i] = {k: vd[k].detach().clone() for k in W_KEYS}
                return out, vd

            layer.forward = wrapped

    # -- context managers -------------------------------------------------
    class _Ctx:
        def __init__(self, ctl, enter, exit_):
            self.ctl, self.enter, self.exit_ = ctl, enter, exit_

        def __enter__(self):
            self.enter()
            return self.ctl

        def __exit__(self, *exc):
            self.exit_()
            return False

    def capturing(self):
        def enter():
            self._captured = {}
            self._capture = True

        def exit_():
            self._capture = False

        return self._Ctx(self, enter, exit_)

    def injecting(self, weights, apply_only=True):
        def enter():
            self._inject = weights
            self._apply_only = apply_only

        def exit_():
            self._inject = None
            self._apply_only = False

        return self._Ctx(self, enter, exit_)

    # -- snapshots --------------------------------------------------------
    def snapshot(self):
        """Per-layer fast weights captured by the last `capturing()` block."""
        if len(self._captured) != len(self.layers):
            raise RuntimeError(
                f"captured {len(self._captured)}/{len(self.layers)} layers; "
                "did the forward run inside capturing()?"
            )
        return [self._captured[i] for i in range(len(self.layers))]

    def initial(self):
        """The slow-weight initialisation each forward starts from."""
        out = []
        for layer in self.layers:
            batch_repeat = 1
            out.append(
                {k: getattr(layer, k).detach().clone().repeat(batch_repeat, 1, 1) for k in W_KEYS}
            )
        return out


def blend(w_a, w_b, alpha):
    """alpha * w_a + (1 - alpha) * w_b, per layer and per matrix."""
    return [
        {k: (alpha * a[k].float() + (1.0 - alpha) * b[k].float()).to(a[k].dtype) for k in W_KEYS}
        for a, b in zip(w_a, w_b)
    ]


def lowrank_delta(w_a, w_init, rank):
    """Rank-r truncation of (w_a - w_init), returned as w_init + P_r(delta).

    This is the intervention the intended method can actually express: a
    low-rank skill written on top of the slow-weight initialisation, rather than
    a wholesale state substitution.
    """
    out = []
    for a, i0 in zip(w_a, w_init):
        entry = {}
        for k in W_KEYS:
            delta = (a[k].float() - i0[k].float())
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            r = min(rank, s.shape[-1])
            approx = (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :]
            entry[k] = (i0[k].float() + approx).to(a[k].dtype)
        out.append(entry)
    return out


def lowrank_correction(w_base, w_target, rank, lam):
    """w_base + lam * P_r(w_target - w_base), per layer and per matrix.

    This is the intervention a skill bank actually performs: keep the state the
    stream has arrived at and add a low-rank correction toward a remembered one.
    Distinct from substituting the remembered state wholesale, which is what a
    blend between two absolute states does.
    """
    out = []
    for base, target in zip(w_base, w_target):
        entry = {}
        for k in W_KEYS:
            diff = target[k].float() - base[k].float()
            if rank is None:
                approx = diff
            else:
                u, s, vh = torch.linalg.svd(diff, full_matrices=False)
                r = min(rank, s.shape[-1])
                approx = (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :]
            entry[k] = (base[k].float() + lam * approx).to(base[k].dtype)
        out.append(entry)
    return out


def diff_spectrum(w_a, w_b, ranks):
    """Singular-value energy of (w_a - w_b), per layer and matrix."""
    rows = []
    for layer, (a, b) in enumerate(zip(w_a, w_b)):
        for k in W_KEYS:
            diff = (a[k].float() - b[k].float()).squeeze(0)
            s = torch.linalg.svdvals(diff)
            energy = (s**2).cumsum(0) / (s**2).sum()
            rows.append({
                "layer": layer,
                "matrix": k,
                "shape": list(diff.shape),
                "energy": {r: energy[min(r, len(s)) - 1].item() for r in ranks},
                "stable_rank": ((s**2).sum() / s[0] ** 2).item(),
                "rel_norm": (torch.linalg.norm(diff) / torch.linalg.norm(b[k].float())).item(),
            })
    return rows


def delta_norms(w, w_init):
    """Per-layer ||W - W_init||_F, for the adaptation-cost profile."""
    stats = collections.OrderedDict()
    for idx, (a, i0) in enumerate(zip(w, w_init)):
        stats[idx] = {
            k: torch.linalg.norm((a[k].float() - i0[k].float()).flatten()).item() for k in W_KEYS
        }
    return stats
