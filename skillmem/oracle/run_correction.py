"""E4 + E5: is the *difference* between two adapted states low-rank, and is it
usable as a correction?

E1 measured a blend between two absolute fast-weight states and found the path
collapses between them.  That says the states sit in different basins; it does
not say the step from one to the other is unusable, because a skill bank never
substitutes a state -- it keeps the state the stream arrived at and adds a
correction on top.

E4  spectrum of D = W_A - W_AB.  The bank stores the difference, not the whole
    adaptation, and differences can be far lower rank than their operands.
E5  render with W_AB + lam * P_r(D), sweeping lam, scoring A's region and B's
    region separately.  The verdict is the slope at lam = 0.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--a-span", type=float, default=0.30)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--lambdas", default="0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    parser.add_argument(
        "--correction-ranks", default="",
        help="ranks to run the lambda sweep at; empty means pick from E4",
    )
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

    from oracle.fastweight_hooks import FastWeightController, diff_spectrum, lowrank_correction
    from oracle.oracle_data import SceneViews, segment_plan

    ctl = FastWeightController(model)
    scene = SceneViews(args.scene, config.model.image_size, config.model.image_size_x)
    plan = segment_plan(
        len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b, a_span=args.a_span
    )
    scene.normalise(sorted(set(sum(plan.values(), []))))
    a, b = plan["a_input"], plan["b_input"]
    targets = plan["a_target"] + plan["b_target"]
    split = len(plan["a_target"])
    print(f"scene {scene.scene_name[:16]}: {len(scene)} frames")

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

    # W is updated over the input tokens only, so W_A is well defined from a
    # run whose virtual cameras cover A alone.
    batch_a = scene.batch(a, a, targets, device=device)
    _, w_a = run(batch_a, len(a), len(a), capture=True)

    batch_ab = scene.batch(a + b, a + b, targets, device=device)
    psnr_base, w_ab = run(batch_ab, len(a + b), len(a + b), capture=True)

    ranks = [int(r) for r in args.ranks.split(",")]
    rows = diff_spectrum(w_a, w_ab, ranks)
    print("\nE4  spectrum of D = W_A - W_AB")
    print(f"{'':>6} " + " ".join(f"{r:>7}" for r in ranks))
    for key in ("w0", "w1", "w2"):
        vals = [np.mean([r_["energy"][r] for r_ in rows if r_["matrix"] == key]) for r in ranks]
        print(f"{key:>6} " + " ".join(f"{v:7.3f}" for v in vals))
    mean_energy = {r: float(np.mean([r_["energy"][r] for r_ in rows])) for r in ranks}
    stable = float(np.mean([r_["stable_rank"] for r_ in rows]))
    rel = float(np.mean([r_["rel_norm"] for r_ in rows]))
    print(f"\nmean stable rank {stable:.1f} of {rows[0]['shape'][0]}   "
          f"mean ||D||/||W_AB|| {rel:.4f}")

    if args.correction_ranks:
        corr_ranks = [None if r == "full" else int(r) for r in args.correction_ranks.split(",")]
    else:
        pick = next((r for r in ranks if mean_energy[r] >= 0.80), ranks[-1])
        corr_ranks = [8, pick, None]
        corr_ranks = list(dict.fromkeys(corr_ranks))
    print(f"E5 will sweep at ranks {corr_ranks} (None = full-rank difference)")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    base_row = {
        "lam": 0.0, "rank": "base",
        "a": summarise(psnr_base[:split]), "b": summarise(psnr_base[split:]),
    }
    print("\nE5  W_AB + lam * P_r(W_A - W_AB)")
    print(f"  baseline            A {base_row['a']['mean']:.3f} dB   B {base_row['b']['mean']:.3f} dB")

    sweep = [base_row]
    for rank in corr_ranks:
        label = "full" if rank is None else f"r{rank}"
        for lam in lambdas:
            if lam == 0.0:
                continue
            psnr, _ = run(
                batch_ab, len(a + b), len(a + b),
                inject=lowrank_correction(w_ab, w_a, rank, lam),
            )
            row = {
                "lam": lam, "rank": label,
                "a": summarise(psnr[:split]), "b": summarise(psnr[split:]),
            }
            sweep.append(row)
            da = row["a"]["mean"] - base_row["a"]["mean"]
            db = row["b"]["mean"] - base_row["b"]["mean"]
            print(f"  {label:>5} lam={lam:<5} A {row['a']['mean']:.3f} ({da:+.3f})   "
                  f"B {row['b']['mean']:.3f} ({db:+.3f})")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{scene.scene_name[:16]}_correction.json")
    json.dump(
        {
            "scene": scene.scene_name, "plan": plan,
            "e4": {"rows": rows, "mean_energy": mean_energy,
                   "stable_rank": stable, "rel_norm": rel},
            "e5": sweep,
        },
        open(path, "w"), indent=2,
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
