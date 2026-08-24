"""Saving and restoring runs.

The bank's buffers -- usage counts and slot ages -- are part of the state, not
bookkeeping. Restoring a run without them resets every slot's age to zero and
the revival machinery immediately reseeds slots that were doing fine.
"""

import os
import shutil
from typing import Optional

import torch


def save(path: str, model, optimizer=None, scheduler=None, step: int = 0,
         best: Optional[float] = None, config=None, is_best: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),          # includes bank usage/age buffers
        "step": step,
        "best": best,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)                     # atomic: a killed job cannot truncate
    if is_best:
        shutil.copyfile(path, os.path.join(os.path.dirname(path), "best.pt"))


def load(path: str, model, optimizer=None, scheduler=None, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    if not strict and (missing or unexpected):
        print(f"checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
