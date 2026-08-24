"""Stable per-Gaussian identity across densification.

gsplat's ``duplicate`` / ``split`` / ``remove`` apply the *same* index permutation
to every tensor stored in the strategy ``state`` dict as they do to the
parameters themselves::

    duplicate: state[k] = cat([v, v[sel]])
    split:     state[k] = cat([v[rest], v[sel].repeat(2, ...)])
    remove:    state[k] = v[~mask]

So any per-Gaussian accumulator placed in ``state`` is carried through
densification for free. This class only has to do the one thing that cannot be
expressed as a permutation: mint fresh unique ids for Gaussians that were newly
created by a grow event, while leaving the *lineage root* id inherited.

Fields maintained here:
  ``uid``        unique id of the live Gaussian (re-minted for new children)
  ``gid``        lineage root id, inherited by every descendant -- this is the
                 ``gid`` that RESEARCH_PLAN.md 6.3 asks for
  ``dens_count`` number of grow events this Gaussian was involved in; 0 selects
                 the "never densified" subpopulation used for the clean analysis
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from gsplat.strategy import DefaultStrategy


class LineageStrategy(DefaultStrategy):
    """:class:`DefaultStrategy` with lineage-stable ids. Densification is unchanged."""

    def init_identity(self, state: Dict[str, Any], n: int, device) -> None:
        state["uid"] = torch.arange(n, dtype=torch.int64, device=device)
        state["gid"] = state["uid"].clone()
        state["dens_count"] = torch.zeros(n, dtype=torch.int32, device=device)
        self._next_uid = n

    def append_identity(self, state: Dict[str, Any], n_new: int, device) -> torch.Tensor:
        """Mint ids for ``n_new`` freshly spawned Gaussians. Returns their uids."""
        new = torch.arange(self._next_uid, self._next_uid + n_new, dtype=torch.int64, device=device)
        self._next_uid += n_new
        state["uid"] = torch.cat([state["uid"], new])
        state["gid"] = torch.cat([state["gid"], new.clone()])
        state["dens_count"] = torch.cat(
            [state["dens_count"], torch.zeros(n_new, dtype=torch.int32, device=device)]
        )
        return new

    # -- densification hook ------------------------------------------------
    def _grow_gs(self, params, optimizers, state, step):
        n_dupli, n_split = super()._grow_gs(params, optimizers, state, step)
        if n_dupli or n_split:
            self._remint(state)
        return n_dupli, n_split

    @torch.no_grad()
    def _remint(self, state: Dict[str, Any]) -> None:
        """After a grow event, a parent's uid appears on each of its children."""
        uid = state["uid"]
        _, inv, cnt = torch.unique(uid, return_inverse=True, return_counts=True)
        shared = cnt[inv] > 1
        if not bool(shared.any()):
            return
        state["dens_count"][shared] += 1

        order = torch.argsort(uid, stable=True)
        su = uid[order]
        is_first = torch.ones_like(su, dtype=torch.bool)
        is_first[1:] = su[1:] != su[:-1]
        rename = order[~is_first]
        n = int(rename.numel())
        state["uid"][rename] = torch.arange(
            self._next_uid, self._next_uid + n, dtype=torch.int64, device=uid.device
        )
        self._next_uid += n
