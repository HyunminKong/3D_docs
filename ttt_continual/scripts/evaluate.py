#!/usr/bin/env python
"""Evaluate a checkpoint, with the bank on and off.

Reporting both is required rather than optional. The headline number for this
project is how much of what was lost the bank returns, and its denominator is
the interference of *this* model without a bank -- training may reduce
interference on its own, which would mean the memory is redundant, and a single
figure cannot tell the two apart.
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt_continual.data.nuscenes import DataConfig, StreamingEpisodes, collate_episode
from ttt_continual.engine import checkpoint
from ttt_continual.engine.trainer import TrainConfig, Trainer
from ttt_continual.losses.reconstruction import LossWeights
from ttt_continual.models.model import ModelConfig, StreamingReconstructor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(open(args.config)) if args.config else payload["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = dict(cfg["model"])
    m["image_size"] = tuple(m["image_size"])
    m["ttt_layers"] = tuple(m["ttt_layers"])
    m["bank_layers"] = tuple(m["bank_layers"])
    model = StreamingReconstructor(ModelConfig(**m)).to(device)
    checkpoint.load(args.checkpoint, model)

    d = dict(cfg["data"]); d["image_size"] = tuple(d["image_size"])
    val = StreamingEpisodes(DataConfig(**d), "val", seed=cfg["seed"] + 1)
    loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_episode)

    t = dict(cfg["train"]); t["loss_weights"] = LossWeights(**t.pop("loss_weights"))
    trainer = Trainer(model, None, None, TrainConfig(**t), None, device)

    bank = model.bank
    with_bank = trainer.evaluate(loader, args.episodes)
    model.bank = None
    without = trainer.evaluate(loader, args.episodes)
    model.bank = bank

    print(f"{'metric':>18} {'bank off':>10} {'bank on':>10} {'delta':>10}")
    for k in sorted(set(with_bank) | set(without)):
        a, b = without.get(k, float('nan')), with_bank.get(k, float('nan'))
        print(f"{k:>18} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

    gap_off = without.get("psnr_current", 0) - without.get("psnr_revisit", 0)
    gap_on = with_bank.get("psnr_current", 0) - with_bank.get("psnr_revisit", 0)
    if gap_off > 0:
        print(f"\ninterference, bank off: {gap_off:.3f} dB   <- the denominator")
        print(f"interference, bank on : {gap_on:.3f} dB")
        print(f"recovery rate         : {(gap_off - gap_on) / gap_off * 100:.1f}%")


if __name__ == "__main__":
    main()
