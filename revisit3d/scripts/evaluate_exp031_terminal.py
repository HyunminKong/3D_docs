#!/usr/bin/env python3
"""Immutable causal reservoir-64 terminal evaluation on EXP-021."""
from __future__ import annotations
import argparse,hashlib,json,random
from dataclasses import replace
from pathlib import Path
import joblib,numpy as np,torch,yaml
from revisit3d.experiments import CachedAtomSegment
from revisit3d.experiments.exp012_minimal import adapt_minimal
from revisit3d.models import SpatialPlasticityHead,visual_transport
from revisit3d.scripts.evaluate_exp009_causal_dino_retrieval import _cpu_atom,_device_atom,_identifier,_timestamp
from revisit3d.scripts.evaluate_exp010_absolute_geometry import LidarProjector,_depth_metrics,_query_lidar
from revisit3d.scripts.fit_exp016_unified_utility_address import _sha256
from revisit3d.scripts.train_exp024_metric_aligned_atom import _query_depth

PRIMARY=("silog","abs_rel","point_epe_m"); METRICS=("silog","abs_rel","rmse_m","delta1","point_epe_m")
def _score(compiled,current,source):
    c=current.numpy().astype(np.float64); s=source.numpy().astype(np.float64)
    return float(compiled["intercept"]+c@compiled["current"]+s@(compiled["source"]+c*compiled["interaction"]))
def _summary(rows,p):
    groups=sorted({r["component"] for r in rows}); return {"targets":len(rows),"components":len(groups),**{
        m:float(np.mean([np.mean([r[p][m] for r in rows if r["component"]==g]) for g in groups])) for m in METRICS}}
def _boot(rows,left,right,m,samples,seed):
    groups=sorted({r["component"] for r in rows}); v=np.asarray([np.mean([r[left][m]-r[right][m] for r in rows if r["component"]==g]) for g in groups]); rng=np.random.default_rng(seed); d=rng.choice(v,(samples,len(v)),replace=True).mean(1)
    return {"direction":f"{left}_minus_{right}_positive_means_{right}_better","mean_improvement":float(v.mean()),"ci95":[float(np.quantile(d,.025)),float(np.quantile(d,.975))]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/EXP-031_terminal_evaluation_v10.yaml"); ap.add_argument("--confirm-terminal-evaluation",action="store_true"); a=ap.parse_args()
    if not a.confirm_terminal_evaluation: raise SystemExit("terminal evaluation requires confirmation")
    cp=Path(a.config); c=yaml.safe_load(cp.read_text()); out=Path(c["output"]["result"])
    if out.exists(): raise RuntimeError("terminal result exists")
    auth=json.loads(Path(c["authorization"]).read_text()); lock=json.loads(Path(c["terminal_lock"]).read_text()); cr=json.loads(Path(c["stage1"]["cache_result"]).read_text())
    atom_path=Path(c["model"]["atom_checkpoint"]); address_path=Path(c["model"]["address_artifact"]); cache_path=Path(c["stage1"]["cache"])
    if not (auth["registered_gate"]["passed"] and not auth["exp021_terminal_accessed"] and cr["cache_sha256"]==_sha256(cache_path)
            and cr["rows"]==214 and cr["model_output_accessed"] and not cr["lidar_decoded"] and lock["terminal_test_locked"]): raise RuntimeError("terminal lock contract failed")
    atom=torch.load(atom_path,map_location="cpu",weights_only=False); address=joblib.load(address_path); geo=torch.load(cache_path,map_location="cpu",weights_only=False,mmap=True); manifest=json.loads(Path(c["data"]["manifest"]).read_text())
    if not (atom["experiment"]=="EXP-028" and address["experiment"]=="EXP-029" and len(geo["rows"])==len(manifest)==214): raise RuntimeError("frozen artifact contract failed")
    contexts,targets={},{}
    for i,row in enumerate(manifest):
        for tag,ct in (("a","a_context"),("b","b_context"),("a_prime","a_prime_context")):
            k=_identifier(row[tag]); contexts.setdefault(k,{"id":k,"segment":row[tag],"cache_index":i,"cache_tag":ct,"location":row["location"]})
        k=_identifier(row["a_prime"]); targets.setdefault(k,{"id":k,"cache_index":i,"component":f"component-{row['component_id']}","location":row["location"],"segment":row["a_prime"]})
    scene_root=Path(c["data"]["scene_root"]); mc={}
    for x in contexts.values(): x["timestamp"]=_timestamp(x["segment"],scene_root,mc)
    dev=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=True; head=SpatialPlasticityHead(feature_dim=int(c["foundation"]["feature_dim"])).to(dev); head.load_state_dict(atom["head"]); head.eval().requires_grad_(False)
    projector=LidarProjector(c["data"]["nuscenes_root"],minimum_depth=float(c["lidar"]["minimum_depth_m"]),maximum_depth=float(c["lidar"]["maximum_depth_m"]),version=c["data"]["nuscenes_version"])
    compiled=address["compiled_mips"]; threshold=float(address["acceptance_threshold"]); cap=int(c["bank"]["capacity"]); strength=float(c["method"]["reuse_strength"]); rows=[]
    with torch.enable_grad():
      for loc in sorted({x["location"] for x in contexts.values()}):
        events=sorted([x for x in contexts.values() if x["location"]==loc],key=lambda x:(x["timestamp"],x["id"])); memory={}; bank=[]; seen=0; rng=random.Random(int(c["seed"])+int(hashlib.sha1(loc.encode()).hexdigest()[:8],16))
        for e in events:
            key=e["id"]; p=geo["rows"][e["cache_index"]]["segments"][e["cache_tag"]]; seg=CachedAtomSegment.from_cache(p,"current" if key in targets else "source",dev); zero=seg.atom(head); code=adapt_minimal(head,seg,zero.code,step_size=float(c["method"]["step_size"])); state={"atom":_cpu_atom(replace(zero,code=code.detach())),"descriptor":zero.key.mean((1,2))[0].detach().cpu()}
            if key in targets and bank:
                target=targets[key]; q=CachedAtomSegment.from_cache(geo["rows"][target["cache_index"]]["segments"]["a_prime_query"],"query",dev); qzero=q.atom(head); current=replace(zero,code=code); pred_cur=_query_depth(head,q,qzero,current); side=pred_cur.shape[-1]; gt,valid=_query_lidar(projector,scene_root,target["segment"],side); intr=q.intrinsics[0].detach().cpu().numpy()
                def metrics(pred): return _depth_metrics(pred.detach().cpu().numpy(),gt,valid,intr,image_size=q.image_size,minimum_cells=int(c["lidar"]["minimum_cells_per_view"]))
                curm=metrics(pred_cur); scores={k:_score(compiled,state["descriptor"],memory[k]["descriptor"]) for k in bank}; winner=max(bank,key=lambda k:(scores[k],k)); take=scores[winner]>threshold; candm={}
                for source_key in bank:
                    src=_device_atom(memory[source_key]["atom"],dev); transported=visual_transport(src,zero).code; candidate=replace(zero,code=(code+strength*transported).clamp(-1,1)); candm[source_key]=metrics(_query_depth(head,q,qzero,candidate))
                if curm is not None and all(v is not None for v in candm.values()):
                    app=max(bank,key=lambda k:float(state["descriptor"]@memory[k]["descriptor"]/max(state["descriptor"].norm()*memory[k]["descriptor"].norm(),1e-12)))
                    full=candm[winner] if take else curm; appearance=candm[app] if take else curm; randomm={m:float(np.mean([candm[k][m] for k in bank])) if take else curm[m] for m in METRICS}; oracle=min(candm.values(),key=lambda x:x["silog"])
                    rows.append({"target":key,"component":target["component"],"location":loc,"accepted":take,"bank_size":len(bank),"current":curm,"full":full,"random":randomm,"appearance":appearance,"oracle":oracle})
            seen+=1
            if len(bank)<cap: bank.append(key); memory[key]=state
            else:
                j=rng.randrange(seen)
                if j<cap: old=bank[j]; del memory[old]; bank[j]=key; memory[key]=state
        print(json.dumps({"location":loc,"events":len(events),"evaluated":sum(r["location"]==loc for r in rows)}),flush=True)
    policies=("current","full","random","appearance","oracle"); summaries={p:_summary(rows,p) for p in policies}; samples=int(c["statistics"]["bootstrap_samples"]); seed=int(c["statistics"]["bootstrap_seed"])
    comparisons={name:{m:_boot(rows,left,"full",m,samples,seed+off+j) for j,m in enumerate(PRIMARY)} for name,left,off in (("full_vs_current","current",0),("full_vs_random","random",10),("full_vs_appearance","appearance",20))}
    minimum=int(c["success"]["minimum_positive_intervals_per_comparison"]); checks={"coverage":len(rows)>=int(c["success"]["minimum_targets"]) and len({r["component"] for r in rows})>=int(c["success"]["minimum_components"])}
    for name in ("current","random","appearance"):
        checks[f"full_better_{name}_all_means"]=all(summaries["full"][m]<summaries[name][m] for m in PRIMARY); checks[f"positive_{name}_intervals"]=sum(comparisons[f"full_vs_{name}"][m]["ci95"][0]>0 for m in PRIMARY)>=minimum
    result={"experiment":"EXP-031","stage":"official_test_terminal_evaluation","protocol_revision":c["protocol_revision"],"config":str(cp),"split":"terminal_test","terminal_accessed":True,"terminal_no_further_tuning":True,
            "manifest_sha256":lock["manifest_sha256"],"atom_sha256":_sha256(atom_path),"address_sha256":_sha256(address_path),"cache_sha256":_sha256(cache_path),"acceptance":float(np.mean([r["accepted"] for r in rows])),
            "summaries":summaries,"comparisons":comparisons,"registered_gate":{"checks":checks,"passed":all(checks.values())},"rows":rows}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps({"output":str(out),"summaries":summaries,"comparisons":comparisons,"gate":result["registered_gate"]}),flush=True)
if __name__=="__main__": main()
