"""Learning-rate schedules.

Warmup then cosine, with one addition: the fast weight's own learning rate is a
trained parameter, and it is put in a group of its own. It sits inside the inner
update, so a step that is fine for the trunk can be far too large here -- the
value is applied thousands of times along a stream, and an outer step that
doubles it changes every one of them.
"""

import math
from typing import List

from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int,
                  min_ratio: float = 0.05) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


def parameter_groups(model, weight_decay: float = 0.05,
                     bank_lr_scale: float = 1.0,
                     inner_lr_scale: float = 0.1) -> List[dict]:
    """Split parameters by what they do, not by name alone.

    Norms, biases and the fast-weight initialisations are excluded from weight
    decay: decaying a fast weight's starting point pulls the whole stream toward
    zero, and decaying a gain has no meaning.
    """
    decay, no_decay, bank, inner = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lr_head" in name:
            inner.append(p)
        elif name.startswith("bank."):
            bank.append(p)
        elif p.ndim < 2 or "_init" in name or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
    ]
    if bank:
        groups.append({"params": bank, "weight_decay": 0.0, "lr_scale": bank_lr_scale})
    if inner:
        groups.append({"params": inner, "weight_decay": 0.0, "lr_scale": inner_lr_scale})
    return [g for g in groups if g["params"]]


def apply_group_scales(optimizer, base_lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = base_lr * g.get("lr_scale", 1.0)
        g["initial_lr"] = g["lr"]
