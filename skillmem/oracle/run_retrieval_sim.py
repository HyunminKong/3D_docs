"""Would retrieval on this descriptor actually pick a useful skill?

Having a descriptor that correlates with interference is not the same as having
one that supports retrieval. Retrieval works only if states that sit close in
descriptor space are also interchangeable -- if borrowing a neighbour's
adaptation costs less than borrowing a stranger's.

So every pair is measured directly. Each unit (a scene at one A/B cut)
contributes its own adapted state; that state is then applied as a correction on
top of every other unit's, restricted to layers 1, 2 and 7 at rank 8 -- the bank
as actually designed, not the full 24-layer state -- and normalised to a common
magnitude so the penalties are comparable across sources.

The question is then whether descriptor distance predicts the penalty, and
whether clustering the descriptor separates compatible states from incompatible
ones. If it does not, a bank can be built and filled and read and still retrieve
nothing worth having.
"""

import argparse
import importlib
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.causal_descriptors import causal_variants  # noqa: E402
from oracle.run_oracle import (  # noqa: E402
    bootstrap_distributed,
    build_config,
    per_view_psnr,
    summarise,
)

BANK_LAYERS = {1, 2, 7}


def project(diff, rank):
    u, s, vh = torch.linalg.svd(diff, full_matrices=False)
    r = min(rank, s.shape[-1])
    return (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :]


def correction(w_ab, w_source, rank, lam, reference_norm):
    """W_AB + lam * (rank-r step toward w_source), on BANK_LAYERS only.

    Every direction is rescaled to `reference_norm` so that a penalty reflects
    where the correction points, not how big it happens to be.
    """
    from oracle.fastweight_hooks import W_KEYS

    out = []
    for idx, base in enumerate(w_ab):
        if idx not in BANK_LAYERS:
            out.append({k: base[k] for k in W_KEYS})
            continue
        entry = {}
        for k in W_KEYS:
            b = base[k].float()
            direction = project(w_source[idx][k].float() - b, rank)
            norm = torch.linalg.norm(direction)
            if norm > 0:
                direction = direction * (reference_norm[idx][k] / norm)
            entry[k] = (b + lam * direction).to(base[k].dtype)
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--n-scenes", type=int, default=10)
    parser.add_argument("--a-spans", default="0.25,0.45")
    parser.add_argument("--n-a", type=int, default=8)
    parser.add_argument("--n-b", type=int, default=8)
    parser.add_argument("--n-target-a", type=int, default=4)
    parser.add_argument("--n-target-b", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument(
        "--within-scene-only", action="store_true",
        help="only inject between cuts of the same scene, holding scene identity fixed",
    )
    parser.add_argument("--out", default="oracle/results/retrieval_sim.json")
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

    from oracle.fastweight_hooks import W_KEYS, FastWeightController
    from oracle.oracle_data import SceneViews, segment_plan

    ctl = FastWeightController(model)
    scene_dirs = sorted(os.listdir(args.processed_dir))[: args.n_scenes]
    spans = [float(s) for s in args.a_spans.split(",")]

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

    def prepare(name, span):
        path = os.path.join(args.processed_dir, name, "opencv_cameras.json")
        scene = SceneViews(path, config.model.image_size, config.model.image_size_x)
        plan = segment_plan(
            len(scene), args.n_a, args.n_b, args.n_target_a, args.n_target_b, a_span=span
        )
        norm = scene.normalise(sorted(set(sum(plan.values(), []))))
        return scene, plan, norm

    # Pass 1: each unit's own adapted state. Only the bank layers are kept, and
    # on host memory -- a full 24-layer snapshot is 648 MB and twenty of them
    # would not fit anywhere useful.
    units = []
    for name, span in itertools.product(scene_dirs, spans):
        try:
            scene, plan, norm = prepare(name, span)
            a, b = plan["a_input"], plan["b_input"]
            targets = plan["a_target"] + plan["b_target"]
            c2w = norm["c2w"]
            desc = causal_variants(
                np.stack([c2w[v].numpy() for v in a]),
                np.stack([c2w[v].numpy() for v in b]),
            )
            _, w_a = run(scene.batch(a, a, targets, device=device),
                         len(a), len(a), len(targets), capture=True)
            bank = {i: {k: w_a[i][k].cpu() for k in W_KEYS} for i in BANK_LAYERS}
            units.append({"id": f"{name[:12]}@{span}", "scene": name, "a_span": span,
                          "desc": desc, "bank": bank})
            print(f"prepared {units[-1]['id']}")
            del w_a
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"skip {name[:12]}@{span} ({type(exc).__name__}: {exc})")

    print(f"\n{len(units)} units -> {len(units)**2} injections")
    pairs = []
    for target in units:
        scene, plan, _ = prepare(target["scene"], target["a_span"])
        a, b = plan["a_input"], plan["b_input"]
        targets = plan["a_target"] + plan["b_target"]
        split = len(plan["a_target"])
        batch_ab = scene.batch(a + b, a + b, targets, device=device)
        psnr_base, w_ab = run(batch_ab, len(a + b), len(a + b), len(targets), capture=True)
        base_a = summarise(psnr_base[:split])["mean"]
        own_bank = target["bank"]
        ref = {
            i: {k: torch.linalg.norm(
                project(own_bank[i][k].to(device).float() - w_ab[i][k].float(), args.rank))
                for k in W_KEYS}
            for i in BANK_LAYERS
        }
        sources = [s for s in units
                   if not args.within_scene_only or s["scene"] == target["scene"]]
        for source in sources:
            src = {i: {k: v.to(device) for k, v in source["bank"][i].items()}
                   for i in BANK_LAYERS}
            psnr, _ = run(
                batch_ab, len(a + b), len(a + b), len(targets),
                inject=correction(w_ab, src, args.rank, args.lam, ref),
            )
            pairs.append({
                "target": target["id"], "source": source["id"],
                "same_scene": target["scene"] == source["scene"],
                "self": target["id"] == source["id"],
                "delta_a": summarise(psnr[:split])["mean"] - base_a,
            })
        own = next(p for p in pairs if p["target"] == target["id"] and p["self"])
        others = [p["delta_a"] for p in pairs
                  if p["target"] == target["id"] and not p["self"]]
        print(f"{target['id']:>18}  base {base_a:6.3f}  self {own['delta_a']:+.3f}   "
              f"others mean {np.mean(others):+.3f}")
        target["base_a"] = base_a
        del w_ab, batch_ab
        torch.cuda.empty_cache()
        json.dump({"units": [{k: u[k] for k in ("id", "scene", "a_span", "desc")} for u in units],
                   "pairs": pairs, "bank_layers": sorted(BANK_LAYERS),
                   "rank": args.rank, "lam": args.lam},
                  open(args.out, "w"), indent=2)

    payload = {
        "units": [{"id": u["id"], "scene": u["scene"], "a_span": u["a_span"],
                   "desc": u["desc"], "base_a": u.get("base_a")} for u in units],
        "pairs": pairs,
        "bank_layers": sorted(BANK_LAYERS), "rank": args.rank, "lam": args.lam,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
