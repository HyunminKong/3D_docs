"""The reusability objective.

Everything else in this codebase is machinery; this is the claim. A fast weight
trained the ordinary way minimises the loss of the chunk in front of it and has
no reason to leave behind anything a later chunk could use -- and measurement
says it does not, with a rank-8 correction toward a remembered state returning
1.6% of what was lost. The proposal is to make reusability part of what the
update is optimised for, and this term is how.

The naive form does not work and the alternatives fail in specific ways:

  reuse alone            collapses to one constant skill applied everywhere,
                         which is a slow weight with extra steps
  penalise transfer      hacked by attaching junk that is orthogonal to the
                         source regime and harmful elsewhere, scoring well on
                         specificity while carrying no information

What is asked for instead is relative: among the skills on offer, the retrieved
one should help most. A constant bank scores zero margin and is ruled out; and
because the alternatives include doing nothing, a skill cannot earn its margin
by making its rivals bad -- identity cannot be made bad. Every slot has to prove
it beats abstaining.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def contrastive_reuse(losses_per_skill: torch.Tensor, chosen: torch.Tensor,
                      temperature: float = 1.0,
                      null_index: int = 0) -> Dict[str, torch.Tensor]:
    """Cross-entropy over candidate skills, ranked by how much each one helped.

    Args:
        losses_per_skill: (b, 1 + K) reconstruction loss obtained with each
            candidate applied. Column `null_index` is the loss with no skill.
        chosen: (b,) index the bank actually retrieved.

    The target is the candidate that turned out best, so the objective pulls
    retrieval toward whichever skill the outcome vindicated. Lower loss is
    better, hence the negation.
    """
    score = -losses_per_skill / temperature
    best = score.argmax(dim=-1)
    ce = F.cross_entropy(score, best, reduction="none")

    with torch.no_grad():
        top2 = score.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean()
        picked_best = (chosen == best).float().mean()
        beats_null = (score.gather(1, chosen.unsqueeze(1)).squeeze(1)
                      > score[:, null_index]).float().mean()

    return {"loss": ce.mean(), "margin": margin,
            "retrieval_accuracy": picked_best, "beats_null": beats_null}


def foreign_penalty(own_loss: torch.Tensor, foreign_loss: torch.Tensor) -> torch.Tensor:
    """How much worse an unrelated skill is than the right one.

    Not optimised -- it is the collapse detector. If training drives every slot
    toward the same thing, a skill from an unrelated episode stops hurting, and
    this number goes to zero while the headline recovery figure keeps improving.
    Watched from the first run rather than diagnosed afterwards.
    """
    return (foreign_loss - own_loss).mean()


def episode_reuse_targets(chunk_losses: List[torch.Tensor],
                          horizon: int = 4) -> Optional[torch.Tensor]:
    """Aggregate the loss of the `horizon` chunks that follow each write.

    A skill is worth storing if what comes after is better, so credit is
    assigned over a window rather than to the next chunk alone. Chunks with less
    than a full horizon remaining are dropped rather than padded, since a short
    window would score them on less evidence than the rest.
    """
    n = len(chunk_losses)
    if n <= horizon:
        return None
    return torch.stack([torch.stack(chunk_losses[i + 1:i + 1 + horizon]).mean()
                        for i in range(n - horizon)])
