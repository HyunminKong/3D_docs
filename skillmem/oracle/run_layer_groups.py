"""How much of the signal survives if the bank stores only some layers?

Single-layer injections put the scene-specific signal at layer 7 in all three
scenes tested, with layers 1-2 secondary and the whole second half of the trunk
contributing nothing.  But the single-layer effects sum to roughly half of what
injecting every layer does, so the layers are not independent and the per-layer
numbers cannot be added up to price a storage decision.

This injects groups directly.  The number that matters for the bank is the ratio
of a group's effect to the all-layer effect: that is the fraction of
discriminative power retained by storing only those layers.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.run_controls import build_direction  # noqa: E402
from oracle.run_layers import inject_subset  # noqa: E402
from oracle.run_oracle import (  # noqa: E402
    bootstrap_distributed,
    build_config,
    per_view_psnr,
    summarise,
)

GROUPS = {
    "L7": [7],
    "L1-2": [1, 2],
    "L1,2,7": [1, 2, 7],
    "L0-3": [0, 1, 2, 3],
    "L0-7": list(range(8)),
    "L0-11": list(range(12)),
    "L12-23": list(range(12, 24)),
    "all": list(range(24)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--foreign-scene", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--out", default="oracle/results/layer_groups.json")
    args = parser.parse_args()

    bootstrap_distributed()
    from utils import sp_support

    sp_support.init_sp_group(sp_size=1)
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = build_config(
        args.config,
        {
            "evaluation": True, "inference": False, "sp_size": 1,
            "model.use_anything": False, "model.act_ckpt": False,
            "model.gaussians.random_ratio": 0.0,
            "model.gaussians.usage_threshold": 0.001,
            "training.torch_compile": False, "training.target_has_input": False,
            "training.perceptual_loss_weight": 0.0, "training.depth_loss_weight": 0.0,
            "training.sample_ar": False, "training.batch_size_per_gpu": 1,
            "training.frame_method": "first_cam",
        },
    )
    module, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module).__dict__[class_name](config).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"], strict=False)
    model.eval()

    from oracle.fastweight_hooks import FastWeightController
    from oracle.oracle_data import SceneViews, segment_plan

    ctl = FastWeightController(model)
    w_init = ctl.initial()

    def load(path):
        scene = SceneViews(path, config.model.image_size, config.model.image_size_x)
        plan = segment_plan(len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b)
        scene.normalise(sorted(set(sum(plan.values(), []))))
        return scene, plan

    fscene, fplan = load(args.foreign_scene)
    results = []

    for scene_path in args.scenes:
        scene, plan = load(scene_path)
        a, b = plan["a_input"], plan["b_input"]
        targets = plan["a_target"] + plan["b_target"]
        split = len(plan["a_target"])

        def run(batch, n_in, n_virt, capture=False, inject=None):
            config.training.num_input_views = n_in
            config.training.num_virtual_views = n_virt
            config.training.num_target_views = len(targets)
            ctx = ctl.capturing() if capture else ctl.injecting(inject, apply_only=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), ctx:
                out = model(batch)
            psnr = per_view_psnr(out.render["render"], out.target["image"])
            snap = ctl.snapshot() if capture else None
            del out
            torch.cuda.empty_cache()
            return psnr, snap

        _, w_a = run(scene.batch(a, a, targets, device=device), len(a), len(a), capture=True)
        batch_ab = scene.batch(a + b, a + b, targets, device=device)
        psnr_base, w_ab = run(batch_ab, len(a + b), len(a + b), capture=True)
        fa = fplan["a_input"]
        ftargets = fplan["a_target"] + fplan["b_target"]
        _, w_foreign = run(
            fscene.batch(fa, fa, ftargets, device=device), len(fa), len(fa), capture=True
        )

        base_a = summarise(psnr_base[:split])["mean"]
        generator = torch.Generator(device=device).manual_seed(0)
        dirs = {
            kind: build_direction(kind, w_ab, w_a, w_init, w_foreign, args.rank, generator)
            for kind in ("foreign", "random")
        }

        print(f"\n{scene.scene_name[:16]}  baseline A {base_a:.3f} dB")
        print(f"{'group':>8} {'layers':>7} {'foreign':>9} {'random':>9} {'retained':>9}")

        rows, full = {}, None
        for name, layers in GROUPS.items():
            entry = {}
            for kind in ("foreign", "random"):
                psnr, _ = run(
                    batch_ab, len(a + b), len(a + b),
                    inject=inject_subset(w_ab, dirs[kind], set(layers), args.lam),
                )
                entry[kind] = summarise(psnr[:split])["mean"] - base_a
            rows[name] = entry
            if name == "all":
                full = entry["foreign"]

        for name, layers in GROUPS.items():
            entry = rows[name]
            share = entry["foreign"] / full if full else float("nan")
            print(f"{name:>8} {len(layers):>7} {entry['foreign']:+9.3f} "
                  f"{entry['random']:+9.3f} {share * 100:8.1f}%")
            results.append({
                "scene": scene.scene_name, "group": name, "n_layers": len(layers),
                "d_a_foreign": entry["foreign"], "d_a_random": entry["random"],
                "retained": share,
            })

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
