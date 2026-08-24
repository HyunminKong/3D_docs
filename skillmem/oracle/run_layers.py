"""Which layers carry the scene-specific part of the adaptation?

A bank that has to store all 24 layers costs 377 MB per 64 slots at rank 8 --
22% of the trunk, which loses the argument against a parameter-matched baseline
before it starts.  If the reusable signal lives in a handful of layers the same
bank costs tens of MB and the argument survives.

Localisation uses the foreign penalty rather than the memory gain.  Injecting a
correction built from an unrelated scene costs 1.04 dB at lambda = 0.2, forty
times the random control, whereas the memory gain is 0.045 dB and would vanish
into rounding once split 24 ways.  A layer that carries scene-specific content
is a layer where a *wrong* correction hurts; that is the same quantity, measured
where it is large.

Each run injects into one layer (or one contiguous block) and leaves the other
23 at the state the stream arrived at, so the reported delta is that layer's
own contribution.
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
from oracle.run_oracle import (  # noqa: E402
    bootstrap_distributed,
    build_config,
    per_view_psnr,
    summarise,
)


def inject_subset(w_ab, directions, layers, lam):
    """Apply `directions` only on `layers`; every other layer keeps W_AB."""
    from oracle.fastweight_hooks import W_KEYS

    out = []
    for idx, base in enumerate(w_ab):
        if idx in layers:
            entry = {
                k: (b + lam * d).to(b.dtype)
                for k, (b, d) in directions[idx].items()
            }
        else:
            entry = {k: base[k] for k in W_KEYS}
        out.append(entry)
    return out


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
    parser.add_argument("--lam-foreign", type=float, default=0.2)
    parser.add_argument("--lam-memory", type=float, default=0.05)
    parser.add_argument("--out", default="oracle/results/layers.json")
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
    n_layers = len(ctl.layers)
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
        base_b = summarise(psnr_base[split:])["mean"]
        generator = torch.Generator(device=device).manual_seed(0)
        dirs = {
            kind: build_direction(kind, w_ab, w_a, w_init, w_foreign, args.rank, generator)
            for kind in ("memory", "foreign", "random")
        }

        print(f"\n{scene.scene_name[:16]}   baseline A {base_a:.3f} dB  B {base_b:.3f} dB")
        print(f"{'layer':>6} {'foreign dA':>11} {'random dA':>11} {'memory dA':>11}")

        rows = []
        for layer in range(n_layers):
            row = {"scene": scene.scene_name, "layer": layer}
            for kind, lam in (
                ("foreign", args.lam_foreign),
                ("random", args.lam_foreign),
                ("memory", args.lam_memory),
            ):
                psnr, _ = run(
                    batch_ab, len(a + b), len(a + b),
                    inject=inject_subset(w_ab, dirs[kind], {layer}, lam),
                )
                row[f"d_a_{kind}"] = summarise(psnr[:split])["mean"] - base_a
                row[f"d_b_{kind}"] = summarise(psnr[split:])["mean"] - base_b
            rows.append(row)
            print(f"{layer:>6} {row['d_a_foreign']:+11.3f} {row['d_a_random']:+11.3f} "
                  f"{row['d_a_memory']:+11.3f}")

        # all layers at once, for reference against the sum of the parts
        for kind, lam in (("foreign", args.lam_foreign), ("memory", args.lam_memory)):
            psnr, _ = run(
                batch_ab, len(a + b), len(a + b),
                inject=inject_subset(w_ab, dirs[kind], set(range(n_layers)), lam),
            )
            total = summarise(psnr[:split])["mean"] - base_a
            part_sum = sum(r[f"d_a_{kind}"] for r in rows)
            print(f"  all layers {kind}: {total:+.3f} dB   (sum of single layers {part_sum:+.3f})")
            results.append({"scene": scene.scene_name, "layer": "all", "kind": kind,
                            "d_a": total, "sum_of_parts": part_sum})

        results.extend(rows)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
