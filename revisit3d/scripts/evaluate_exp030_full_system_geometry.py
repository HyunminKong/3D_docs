#!/usr/bin/env python3
"""No-fit geometry audit of the frozen EXP-028 atom plus EXP-029 address."""
from __future__ import annotations

import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch, yaml

from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.models import SpatialPlasticityHead, visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _device_atom
from revisit3d.scripts.fit_exp016_unified_utility_address import _context_tables, _sha256, _strict_oof
from revisit3d.scripts.train_exp024_metric_aligned_atom import (
    METRICS, _lidar_cache, _numpy_metrics, _query_depth, _risk,
)

PRIMARY = ("silog", "abs_rel", "point_epe_m")

def _cpu(atom):
    return type(atom)(*(x.detach().cpu() for x in (atom.xyz, atom.scale, atom.key, atom.code, atom.confidence)))

def _summary(rows, policy):
    groups = sorted({r["component"] for r in rows})
    return {"targets": len(rows), "components": len(groups), **{
        m: float(np.mean([np.mean([r[policy][m] for r in rows if r["component"] == g]) for g in groups]))
        for m in METRICS}}

def _boot(rows, left, right, metric, samples, seed):
    groups = sorted({r["component"] for r in rows})
    values = np.asarray([np.mean([r[left][metric] - r[right][metric] for r in rows if r["component"] == g]) for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (samples, len(values)), replace=True).mean(1)
    return {"direction": f"{left}_minus_{right}_positive_means_{right}_better", "mean_improvement": float(values.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/EXP-030_full_system_geometry_audit_v10.yaml")
    args = ap.parse_args(); cp = Path(args.config); c = yaml.safe_load(cp.read_text()); out = Path(c["output"]["result"])
    if out.exists() or not torch.cuda.is_available() or c["data"]["split"] != "train": raise RuntimeError("EXP-030 contract failed")
    prior = json.loads(Path(c["prior_stage"]).read_text()); pair_path = Path(c["data"]["candidate_cache"])
    atom_path = Path(c["model"]["atom_checkpoint"]); address_path = Path(c["model"]["address_artifact"])
    if not (prior["experiment"] == "EXP-029" and prior["registered_gate"]["passed"]
            and prior["candidate_cache_sha256"] == _sha256(pair_path) and prior["artifact_sha256"] == _sha256(address_path)
            and not prior["validation_accessed"] and not prior["test_accessed"]): raise RuntimeError("EXP-029 frozen contract failed")
    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    matrix = pair["features"].numpy().astype(np.float64); utility = pair["utility"].numpy().astype(np.float64)
    meta, target_table = pair["metadata"], pair["target_table"]
    tl = np.asarray([r["target_location"] for r in meta]); sl = np.asarray([r["source_location"] for r in meta])
    pred, folds = _strict_oof(matrix, utility, tl, sl, float(c["address"]["ridge_alpha"]))
    manifest = json.loads(Path(c["data"]["manifest"]).read_text())
    geo = torch.load(c["data"]["geometry_cache"], map_location="cpu", weights_only=False, mmap=True)
    ckpt = torch.load(atom_path, map_location="cpu", weights_only=False)
    if not (ckpt["experiment"] == "EXP-028" and not ckpt["exp021_terminal_accessed"]): raise RuntimeError("atom contract failed")
    context, targets, _ = _context_tables(manifest); dev = torch.device("cuda")
    head = SpatialPlasticityHead(feature_dim=int(c["model"]["feature_dim"])).to(dev); head.load_state_dict(ckpt["head"])
    head.eval().requires_grad_(False); lidar = _lidar_cache(manifest, geo, c, dev)
    needed = {r["target_context"] for r in meta} | {r["source_context"] for r in meta}; atoms = {}
    with torch.enable_grad():
        for n, key in enumerate(sorted(needed), 1):
            info = context[key]; p = geo["rows"][info["cache_index"]]["segments"][info["cache_tag"]]
            seg = CachedAtomSegment.from_cache(p, "current" if key in targets else "source", dev); zero = seg.atom(head)
            code = adapt_minimal(head, seg, zero.code, step_size=float(c["method"]["step_size"])); atoms[key] = _cpu(replace(zero, code=code))
            if n % 100 == 0 or n == len(needed): print(json.dumps({"atoms": n, "total": len(needed)}), flush=True)
    by_ep = {}
    for i, r in enumerate(meta): by_ep.setdefault(r["episode"], []).append(i)
    rows, max_err = [], 0.0; threshold = float(c["method"]["acceptance_threshold"])
    with torch.enable_grad():
        for n, ep in enumerate(sorted(target_table), 1):
            ids = by_ep.get(ep, [])
            if not ids: continue
            key = meta[ids[0]]["target_context"]; target = targets[key]; idx = target["cache_index"]
            cur_atom = _device_atom(atoms[key], dev); qp = geo["rows"][idx]["segments"]["a_prime_query"]
            query = CachedAtomSegment.from_cache(qp, "query", dev); qzero = query.atom(head); gt, valid = lidar[idx]
            cur_pred = _query_depth(head, query, qzero, cur_atom); cur_risk = _risk(cur_pred, gt, valid, c)
            cur_metrics = _numpy_metrics(cur_pred, gt, valid, query, c); cms, risks = {}, {}
            for i in ids:
                src = _device_atom(atoms[meta[i]["source_context"]], dev); transported = visual_transport(src, cur_atom).code
                cand = replace(cur_atom, code=(cur_atom.code + float(c["method"]["reuse_strength"]) * transported).clamp(-1, 1))
                p = _query_depth(head, query, qzero, cand); risks[i] = _risk(p, gt, valid, c); cms[i] = _numpy_metrics(p, gt, valid, query, c)
                max_err = max(max_err, abs((cur_risk-risks[i])/max(abs(cur_risk),1e-8)-utility[i]))
            winner = max(ids, key=lambda i: (pred[i], meta[i]["source_context"])); take = bool(pred[winner] > threshold)
            current_descriptor = matrix[ids[0], :64]
            app = max(ids, key=lambda i: float(current_descriptor @ matrix[i,64:128] / max(np.linalg.norm(current_descriptor)*np.linalg.norm(matrix[i,64:128]),1e-12)))
            oracle = min(ids, key=lambda i: risks[i]); full = cms[winner] if take else cur_metrics; appearance = cms[app] if take else cur_metrics
            random = {m: float(np.mean([cms[i][m] for i in ids])) if take else cur_metrics[m] for m in METRICS}
            oracle_metrics = cms[oracle] if risks[oracle] < cur_risk else cur_metrics
            rows.append({"episode":ep,"component":target_table[ep]["component"],"location":target_table[ep]["location"],"accepted":take,
                         "current":cur_metrics,"full":full,"random":random,"appearance":appearance,"oracle":oracle_metrics})
            if n % 25 == 0 or n == len(target_table): print(json.dumps({"targets":n,"total":len(target_table)}),flush=True)
    if max_err > 1e-5: raise RuntimeError(f"utility replay mismatch {max_err}")
    policies=("current","full","random","appearance","oracle"); summaries={p:_summary(rows,p) for p in policies}
    samples=int(c["statistics"]["bootstrap_samples"]); seed=int(c["statistics"]["bootstrap_seed"])
    comparisons={name:{m:_boot(rows,left,"full",m,samples,seed+off+j) for j,m in enumerate(PRIMARY)}
                 for name,left,off in (("full_vs_current","current",0),("full_vs_random","random",10),("full_vs_appearance","appearance",20))}
    minimum=int(c["success"]["minimum_positive_intervals_per_comparison"])
    checks={"coverage":len(rows)>=int(c["success"]["minimum_targets"]) and len({r["component"] for r in rows})>=int(c["success"]["minimum_components"])}
    for name,left in (("current","current"),("random","random"),("appearance","appearance")):
        checks[f"full_better_{name}_all_means"] = all(summaries["full"][m] < summaries[left][m] for m in PRIMARY)
        checks[f"positive_{name}_intervals"] = sum(comparisons[f"full_vs_{name}"][m]["ci95"][0] > 0 for m in PRIMARY) >= minimum
    result={"experiment":"EXP-030","stage":"no_fit_full_system_geometry_audit","protocol_revision":c["protocol_revision"],"config":str(cp),
            "split":"train","no_model_fit":True,"parameter_updates":0,"atom_checkpoint_sha256":_sha256(atom_path),
            "address_artifact_sha256":_sha256(address_path),"candidate_cache_sha256":_sha256(pair_path),"address_prediction":"source_safe_location_oof",
            "maximum_utility_replay_error":max_err,"acceptance":float(np.mean([r["accepted"] for r in rows])),"summaries":summaries,
            "comparisons":comparisons,"registered_gate":{"checks":checks,"passed":all(checks.values())},"rows":rows,"folds":folds,
            "validation_accessed":False,"test_accessed":False,"exp021_terminal_accessed":False}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({"output":str(out),"summaries":summaries,"comparisons":comparisons,"gate":result["registered_gate"]}),flush=True)

if __name__ == "__main__": main()
