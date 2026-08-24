"""Is the small gain from a rank-8 correction actually memory?

E5 found that W_AB + 0.05 * P_8(W_A - W_AB) improves both A's region and B's
region by about +0.05 dB.  A correction carrying A's past adaptation should help
A more than B; helping B slightly more is the signature of something generic.
Three controls separate the explanations, all at matched rank and lambda:

memory    P_8(W_A  - W_AB)   the claim
shrink    P_8(W_init- W_AB)  undoing adaptation, i.e. plain regularisation
random    a random rank-8 direction, Frobenius-matched to the memory one
foreign   P_8(W_C  - W_AB)   the same construction from an unrelated scene

If memory is indistinguishable from shrink, random or foreign, the gain carries
no scene-specific information and E5 is not evidence of reuse.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.run_oracle import (  # noqa: E402
    bootstrap_distributed,
    build_config,
    per_view_psnr,
    summarise,
)


def project(diff, rank):
    u, s, vh = torch.linalg.svd(diff, full_matrices=False)
    r = min(rank, s.shape[-1])
    return (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :]


def build_direction(kind, w_ab, w_a, w_init, w_foreign, rank, generator):
    """Per-layer, per-matrix update direction, all Frobenius-matched to `memory`."""
    from oracle.fastweight_hooks import W_KEYS

    out = []
    for idx, base in enumerate(w_ab):
        entry = {}
        for k in W_KEYS:
            b = base[k].float()
            reference = project(w_a[idx][k].float() - b, rank)
            scale = torch.linalg.norm(reference)
            if kind == "memory":
                direction = reference
            elif kind == "shrink":
                direction = project(w_init[idx][k].float() - b, rank)
            elif kind == "foreign":
                direction = project(w_foreign[idx][k].float() - b, rank)
            elif kind == "random":
                m, n = b.shape[-2], b.shape[-1]
                u = torch.randn(b.shape[0], m, rank, device=b.device, generator=generator)
                v = torch.randn(b.shape[0], rank, n, device=b.device, generator=generator)
                direction = u @ v
            else:
                raise ValueError(kind)
            norm = torch.linalg.norm(direction)
            if norm > 0:
                direction = direction * (scale / norm)
            entry[k] = (b, direction)
        out.append(entry)
    return out


def apply_direction(directions, lam):
    return [
        {k: (base + lam * direction).to(base.dtype) for k, (base, direction) in entry.items()}
        for entry in directions
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--foreign-scene", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambdas", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--out", default="oracle/results")
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

    def load(path):
        scene = SceneViews(path, config.model.image_size, config.model.image_size_x)
        plan = segment_plan(len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b)
        scene.normalise(sorted(set(sum(plan.values(), []))))
        return scene, plan

    def run(batch, n_in, n_virt, n_tgt, capture=False, inject=None):
        config.training.num_input_views = n_in
        config.training.num_virtual_views = n_virt
        config.training.num_target_views = n_tgt
        ctx = ctl.capturing() if capture else ctl.injecting(inject, apply_only=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), ctx:
            out = model(batch)
        psnr = per_view_psnr(out.render["render"], out.target["image"])
        snap = ctl.snapshot() if capture else None
        del out
        torch.cuda.empty_cache()
        return psnr, snap

    scene, plan = load(args.scene)
    a, b = plan["a_input"], plan["b_input"]
    targets = plan["a_target"] + plan["b_target"]
    split = len(plan["a_target"])

    _, w_a = run(scene.batch(a, a, targets, device=device), len(a), len(a), len(targets), capture=True)
    batch_ab = scene.batch(a + b, a + b, targets, device=device)
    psnr_base, w_ab = run(batch_ab, len(a + b), len(a + b), len(targets), capture=True)
    w_init = ctl.initial()

    fscene, fplan = load(args.foreign_scene)
    fa = fplan["a_input"]
    ftargets = fplan["a_target"] + fplan["b_target"]
    _, w_foreign = run(
        fscene.batch(fa, fa, ftargets, device=device), len(fa), len(fa), len(ftargets), capture=True
    )

    base_a, base_b = summarise(psnr_base[:split]), summarise(psnr_base[split:])
    print(f"scene {scene.scene_name[:16]}  foreign {fscene.scene_name[:16]}  rank {args.rank}")
    print(f"baseline   A {base_a['mean']:.3f} dB   B {base_b['mean']:.3f} dB\n")
    print(f"{'kind':>8} {'lam':>6} {'A':>9} {'dA':>8} {'B':>9} {'dB':>8}")

    generator = torch.Generator(device=device).manual_seed(0)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = []
    for kind in ("memory", "shrink", "random", "foreign"):
        directions = build_direction(kind, w_ab, w_a, w_init, w_foreign, args.rank, generator)
        for lam in lambdas:
            psnr, _ = run(
                batch_ab, len(a + b), len(a + b), len(targets),
                inject=apply_direction(directions, lam),
            )
            sa, sb = summarise(psnr[:split]), summarise(psnr[split:])
            da, db = sa["mean"] - base_a["mean"], sb["mean"] - base_b["mean"]
            rows.append({"kind": kind, "lam": lam, "a": sa, "b": sb, "da": da, "db": db})
            print(f"{kind:>8} {lam:>6} {sa['mean']:9.3f} {da:+8.3f} {sb['mean']:9.3f} {db:+8.3f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{scene.scene_name[:16]}_controls.json")
    json.dump(
        {"scene": scene.scene_name, "foreign": fscene.scene_name, "rank": args.rank,
         "baseline": {"a": base_a, "b": base_b}, "rows": rows},
        open(path, "w"), indent=2,
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
