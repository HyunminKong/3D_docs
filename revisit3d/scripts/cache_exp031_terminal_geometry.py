#!/usr/bin/env python3
"""One-shot frozen foundation/tracker cache for EXP-031."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,yaml
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.scripts.cache_exp006_frozen_outputs import _geometry_pass,_tracker_pass
from revisit3d.scripts.build_exp021_terminal_manifest import _sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/EXP-031_terminal_evaluation_v10.yaml"); ap.add_argument("--confirm-terminal-access",action="store_true"); a=ap.parse_args()
    if not a.confirm_terminal_access: raise SystemExit("terminal RGB/model access requires confirmation")
    c=yaml.safe_load(Path(a.config).read_text()); output=Path(c["stage1"]["cache"]); result=Path(c["stage1"]["cache_result"])
    if output.exists() or result.exists(): raise RuntimeError("terminal cache output exists")
    conversion=json.loads(Path(c["stage1"]["scene_conversion_result"]).read_text()); lock=json.loads(Path(c["terminal_lock"]).read_text())
    if not (conversion["terminal_access_started"] and conversion["scenes"]==96 and not conversion["image_decoded"]
            and conversion["manifest_sha256"]==lock["manifest_sha256"]): raise RuntimeError("terminal conversion contract failed")
    ds=RevisitEpisodeDataset(c["data"]["manifest"],c["data"]["scene_root"],split=c["data"]["split"],image_size=(c["data"]["image_height"],c["data"]["image_width"]))
    if len(ds)!=214: raise RuntimeError("terminal dataset count changed")
    dev=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=True; rows=_geometry_pass(ds,c,dev); _tracker_pass(ds,rows,c,dev)
    source=Path(c["stage1"]["pca_source_cache"]); pca=torch.load(source,map_location="cpu",weights_only=False,mmap=True)
    if pca.get("split")!="train" or pca.get("protocol_revision")!="v1.5": raise RuntimeError("train PCA contract failed")
    payload={"experiment":"EXP-031","protocol_revision":c["protocol_revision"],"split":c["data"]["split"],"terminal_manifest_sha256":lock["manifest_sha256"],
             "pca_fit_split":"train","pca_source_cache":str(source),"pca_mean":pca["pca_mean"],"pca_components":pca["pca_components"],"rows":rows}
    output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,output)
    summary={"experiment":"EXP-031","stage":"terminal_frozen_cache","protocol_revision":c["protocol_revision"],"cache":str(output),"cache_sha256":_sha256(output),
             "rows":len(rows),"split":c["data"]["split"],"image_decoded":True,"model_output_accessed":True,"lidar_decoded":False,"pca_fit_split":"train"}
    result.parent.mkdir(parents=True,exist_ok=True); result.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))
if __name__=="__main__": main()
