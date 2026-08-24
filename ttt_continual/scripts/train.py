#!/usr/bin/env python
"""Entry point for training."""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttt_continual.data.nuscenes import DataConfig, StreamingEpisodes, collate_episode
from ttt_continual.engine import checkpoint
from ttt_continual.engine.scheduler import apply_group_scales, parameter_groups, warmup_cosine
from ttt_continual.engine.trainer import TrainConfig, Trainer
from ttt_continual.losses.reconstruction import LossWeights
from ttt_continual.models.model import ModelConfig, StreamingReconstructor
from ttt_continual.utils.logging import RunLogger
from ttt_continual.utils.seed import seed_everything, worker_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted key=value pairs, e.g. train.lr=1e-4")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for item in args.override:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(value)

    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    m = dict(cfg["model"])
    m["image_size"] = tuple(m["image_size"])
    m["ttt_layers"] = tuple(m["ttt_layers"])
    m["bank_layers"] = tuple(m["bank_layers"])
    model = StreamingReconstructor(ModelConfig(**m)).to(device)
    print("parameters:", {k: f"{v/1e6:.2f}M" for k, v in model.n_parameters().items()})

    d = dict(cfg["data"]); d["image_size"] = tuple(d["image_size"])
    data_cfg = DataConfig(**d)
    train_set = StreamingEpisodes(data_cfg, "train", seed=cfg["seed"])
    val_set = StreamingEpisodes(data_cfg, "val", seed=cfg["seed"] + 1)
    make = lambda ds, shuffle: DataLoader(
        ds, batch_size=1, shuffle=shuffle, num_workers=4, collate_fn=collate_episode,
        worker_init_fn=worker_init, persistent_workers=True, pin_memory=True)
    train_loader, val_loader = make(train_set, True), make(val_set, False)
    print(f"episodes: {len(train_set)} train / {len(val_set)} val")

    t = dict(cfg["train"])
    t["loss_weights"] = LossWeights(**t.pop("loss_weights"))
    train_cfg = TrainConfig(**t)

    groups = parameter_groups(model, train_cfg.weight_decay)
    opt = torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(0.9, 0.95))
    apply_group_scales(opt, train_cfg.lr)
    sched = warmup_cosine(opt, train_cfg.warmup_steps, train_cfg.total_steps)

    logger = RunLogger(cfg["out_dir"], print_every=train_cfg.log_every)
    trainer = Trainer(model, opt, sched, train_cfg, logger, device)

    start, best = 0, None
    if args.resume:
        payload = checkpoint.load(args.resume, model, opt, sched)
        start, best = payload["step"], payload.get("best")
        trainer.step = start
        print(f"resumed from {args.resume} at step {start}")

    ckpt_dir = os.path.join(cfg["out_dir"], "checkpoints")
    it = iter(train_loader)
    for step in range(start, train_cfg.total_steps):
        trainer.step = step
        stats = {}
        for _ in range(train_cfg.grad_accum):
            try:
                episode = next(it)
            except StopIteration:
                it = iter(train_loader)
                episode = next(it)
            stats = trainer.train_episode(episode)
        stats["grad_norm"] = trainer.optimiser_step()
        stats["lr"] = opt.param_groups[0]["lr"]
        logger.log(step, stats)

        if step and step % train_cfg.eval_every == 0:
            val = trainer.evaluate(val_loader)
            logger.log(step, val, prefix="val_")
            score = val.get("psnr_revisit", val.get("psnr_current", 0.0))
            is_best = best is None or score > best
            best = score if is_best else best
            checkpoint.save(os.path.join(ckpt_dir, "last.pt"), model, opt, sched,
                            step, best, cfg, is_best)
        elif step and step % train_cfg.save_every == 0:
            checkpoint.save(os.path.join(ckpt_dir, "last.pt"), model, opt, sched,
                            step, best, cfg)

    logger.close()


if __name__ == "__main__":
    main()
