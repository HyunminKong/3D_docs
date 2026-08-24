"""How low-rank is the fast-weight adaptation?

The intended skill bank stores rank-r factors, so its whole premise is that the
adaptation a chunk performs lives in a few directions.  This measures that
directly: capture W after streaming a segment, subtract the slow-weight
initialisation, and report how much of ||dW||_F^2 the top-r singular directions
account for, per layer.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.run_oracle import bootstrap_distributed, build_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dl3dv_ar.yaml")
    parser.add_argument("--ckpt", default="checkpoints/dl3dv_ar.pt")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--n-views", type=int, default=16)
    parser.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--out", default="oracle/results")
    args = parser.parse_args()

    bootstrap_distributed()
    from utils import sp_support

    sp_support.init_sp_group(sp_size=1)
    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    import importlib

    config = build_config(
        args.config,
        {
            "evaluation": True, "inference": False, "sp_size": 1,
            "model.use_anything": False, "model.act_ckpt": False,
            "model.gaussians.random_ratio": 0.0,
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
    from oracle.oracle_data import SceneViews

    ctl = FastWeightController(model)
    scene = SceneViews(args.scene, config.model.image_size, config.model.image_size_x)
    step = max(1, len(scene) // (args.n_views + 4))
    views = list(range(0, len(scene), step))[: args.n_views]
    targets = [v + 1 for v in views[:4]]
    scene.normalise(sorted(set(views + targets)))
    batch = scene.batch(views, views, targets, device=device)

    config.training.num_input_views = len(views)
    config.training.num_virtual_views = len(views)
    config.training.num_target_views = len(targets)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), ctl.capturing():
        model(batch)
    w = ctl.snapshot()
    w_init = ctl.initial()

    ranks = [int(r) for r in args.ranks.split(",")]
    rows = []
    for layer, (a, i0) in enumerate(zip(w, w_init)):
        for key in W_KEYS:
            delta = (a[key].float() - i0[key].float()).squeeze(0)
            s = torch.linalg.svdvals(delta)
            energy = (s**2).cumsum(0) / (s**2).sum()
            rows.append({
                "layer": layer, "matrix": key,
                "shape": list(delta.shape),
                "rel_delta": (torch.linalg.norm(delta) / torch.linalg.norm(i0[key].float())).item(),
                "energy": {r: energy[min(r, len(s)) - 1].item() for r in ranks},
                "stable_rank": ((s**2).sum() / s[0] ** 2).item(),
            })

    print(f"{'rank':>6} " + " ".join(f"{r:>7}" for r in ranks))
    for key in W_KEYS:
        vals = [np.mean([row["energy"][r] for row in rows if row["matrix"] == key]) for r in ranks]
        print(f"{key:>6} " + " ".join(f"{v:7.3f}" for v in vals))
    sr = np.mean([row["stable_rank"] for row in rows])
    rd = np.mean([row["rel_delta"] for row in rows])
    print(f"\nmean stable rank {sr:.1f} of {rows[0]['shape'][0]}   mean ||dW||/||W_init|| {rd:.4f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "delta_spectrum.json")
    json.dump({"rows": rows, "ranks": ranks}, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
