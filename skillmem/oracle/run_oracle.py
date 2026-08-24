"""Day-1 oracle: is a past fast-weight state worth anything on revisit?

The question is whether the adaptation tttLRM performs over an early segment of
a trajectory retains value once the model has streamed through a later segment,
or whether it is simply overwritten.  Nothing here is a proposed method: it
measures the ceiling that any retrieval scheme would have to live under, using
perfect information (the exact past state, no descriptor, no retrieval, no
compression).

Because `SelfAttention` is intra-image and the MLP is pointwise, the fast weight
is the only channel from one view to another.  Two consequences the experiment
leans on:

* adding segment B to the input can change A's reconstruction *only* through
  the fast weight, and
* re-rendering with W injected at apply time and alpha = 1 must reproduce the
  A-only condition exactly.  That equality is checked and reported; if it fails,
  the plumbing leaks and no number below should be believed.

E1  interference + blend curve
    virtual cameras and targets are held fixed at A; only the input sequence
    (A vs A+B) and the injected W vary.

E2  trade-off
    input and virtual cameras span A and B; sweeping alpha shows what A gains
    and B loses, which is the question the real method has to answer.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bootstrap_distributed():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.distributed.init_process_group(backend="nccl")


def build_config(config_path, overrides):
    import omegaconf
    from easydict import EasyDict as edict

    config = omegaconf.OmegaConf.load(config_path)
    config = edict(omegaconf.OmegaConf.to_container(config, resolve=True))
    for key, value in overrides.items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return config


def per_view_psnr(render, target):
    """PSNR per target view, matching the model's own convention."""
    mse = ((render.float() - target.float()) ** 2).flatten(2).mean(-1)
    return (-10.0 * torch.log10(mse.clamp(min=1e-12))).squeeze(0).tolist()


def summarise(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--scene", required=True, help="path to opencv_cameras.json")
    parser.add_argument("--n-a", type=int, default=8, help="input views in segment A")
    parser.add_argument("--n-b", type=int, default=8, help="input views in segment B")
    parser.add_argument("--n-target-a", type=int, default=6)
    parser.add_argument("--n-target-b", type=int, default=6)
    parser.add_argument("--a-span", type=float, default=0.30)
    parser.add_argument(
        "--alphas", default="0,0.05,0.1,0.2,0.35,0.5,0.75,1.0",
        help="blend weights on the A-state; the slope at alpha=0 is the verdict",
    )
    parser.add_argument("--ranks", default="", help="optional low-rank truncations, e.g. 1,2,4,8")
    parser.add_argument("--skip-e2", action="store_true")
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
    torch.backends.cudnn.allow_tf32 = True

    config = build_config(
        args.config,
        {
            "evaluation": True,
            "inference": False,
            "sp_size": 1,
            "model.use_anything": False,
            "model.act_ckpt": False,
            "model.gaussians.random_ratio": 0.0,   # keep pruning deterministic
            "model.gaussians.usage_threshold": 0.001,
            "training.torch_compile": False,       # forwards are monkey-patched
            "training.target_has_input": False,
            "training.perceptual_loss_weight": 0.0,
            "training.depth_loss_weight": 0.0,     # skips the depth_anything branch
            "training.sample_ar": False,
            "training.batch_size_per_gpu": 1,
            "training.frame_method": "first_cam",
        },
    )

    module, class_name = config.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module).__dict__[class_name](config).to(device)
    state = torch.load(args.ckpt, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {args.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    from oracle.fastweight_hooks import FastWeightController, blend, delta_norms, lowrank_delta
    from oracle.oracle_data import SceneViews, segment_plan

    ctl = FastWeightController(model)
    print(f"fast-weight layers: {len(ctl.layers)}")

    scene = SceneViews(args.scene, config.model.image_size, config.model.image_size_x)
    plan = segment_plan(
        len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b, a_span=args.a_span
    )
    universe = sorted(set(sum(plan.values(), [])))
    scene.normalise(universe)
    print(f"scene {scene.scene_name}: {len(scene)} frames")
    for key, views in plan.items():
        print(f"  {key:9s} {views}")

    alphas = [float(a) for a in args.alphas.split(",") if a != ""]
    ranks = [int(r) for r in args.ranks.split(",") if r != ""]
    results = {"scene": scene.scene_name, "plan": plan, "alphas": alphas}

    def run(batch, num_input, num_virtual, num_target, capture=False, inject=None):
        config.training.num_input_views = num_input
        config.training.num_virtual_views = num_virtual
        config.training.num_target_views = num_target
        ctx = ctl.capturing() if capture else ctl.injecting(inject, apply_only=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), ctx:
            out = model(batch)
        psnr = per_view_psnr(out.render["render"], out.target["image"])
        snapshot = ctl.snapshot() if capture else None
        del out
        torch.cuda.empty_cache()
        return psnr, snapshot

    # ---- E1: interference and the blend curve --------------------------
    a, b = plan["a_input"], plan["b_input"]
    ta = plan["a_target"]

    batch_a = scene.batch(a, a, ta, device=device)
    psnr_i, w_a = run(batch_a, len(a), len(a), len(ta), capture=True)

    batch_ab = scene.batch(a + b, a, ta, device=device)
    psnr_ii, w_ab = run(batch_ab, len(a + b), len(a), len(ta), capture=True)

    print("\nE1  targets fixed on A's region")
    print(f"  (i)  input A      : {summarise(psnr_i)['mean']:.3f} dB")
    print(f"  (ii) input A+B    : {summarise(psnr_ii)['mean']:.3f} dB")
    print(f"  interference      : {summarise(psnr_i)['mean'] - summarise(psnr_ii)['mean']:+.3f} dB")

    curve = []
    for alpha in alphas:
        psnr, _ = run(batch_ab, len(a + b), len(a), len(ta), inject=blend(w_a, w_ab, alpha))
        curve.append({"alpha": alpha, "a_region": summarise(psnr), "per_view": psnr})
        print(f"  alpha={alpha:<5} A-region {summarise(psnr)['mean']:.3f} dB")

    at_one = [c for c in curve if c["alpha"] == 1.0]
    identity_gap = None
    if at_one:
        identity_gap = at_one[0]["a_region"]["mean"] - summarise(psnr_i)["mean"]
        verdict = "OK" if abs(identity_gap) < 0.05 else "LEAK"
        print(f"  plumbing check (alpha=1 vs condition i): {identity_gap:+.4f} dB  [{verdict}]")

    results["e1"] = {
        "cond_i_a_only": summarise(psnr_i),
        "cond_ii_a_plus_b": summarise(psnr_ii),
        "interference_db": summarise(psnr_i)["mean"] - summarise(psnr_ii)["mean"],
        "curve": curve,
        "identity_gap_db": identity_gap,
    }

    w_init = ctl.initial()
    results["delta_norms"] = {
        "w_a": delta_norms(w_a, w_init),
        "w_ab": delta_norms(w_ab, w_init),
    }

    if ranks:
        lr_curve = []
        for rank in ranks:
            injected = lowrank_delta(w_a, w_init, rank)
            psnr, _ = run(batch_ab, len(a + b), len(a), len(ta), inject=injected)
            lr_curve.append({"rank": rank, "a_region": summarise(psnr)})
            print(f"  rank-{rank:<3} A-region {summarise(psnr)['mean']:.3f} dB")
        results["e1"]["lowrank"] = lr_curve

    # ---- E2: what A gains, what B loses --------------------------------
    if not args.skip_e2:
        targets = ta + plan["b_target"]
        batch_full = scene.batch(a + b, a + b, targets, device=device)
        psnr_full, w_full = run(batch_full, len(a + b), len(a + b), len(targets), capture=True)
        split = len(ta)
        tradeoff = [{
            "alpha": None,
            "a_region": summarise(psnr_full[:split]),
            "b_region": summarise(psnr_full[split:]),
        }]
        print("\nE2  virtual cameras span A and B")
        print(f"  baseline          A {tradeoff[0]['a_region']['mean']:.3f} dB   "
              f"B {tradeoff[0]['b_region']['mean']:.3f} dB")
        for alpha in alphas:
            psnr, _ = run(
                batch_full, len(a + b), len(a + b), len(targets),
                inject=blend(w_a, w_full, alpha),
            )
            row = {
                "alpha": alpha,
                "a_region": summarise(psnr[:split]),
                "b_region": summarise(psnr[split:]),
            }
            tradeoff.append(row)
            print(f"  alpha={alpha:<5}       A {row['a_region']['mean']:.3f} dB   "
                  f"B {row['b_region']['mean']:.3f} dB")
        results["e2"] = tradeoff

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{scene.scene_name[:16]}_oracle.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
