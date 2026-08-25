#!/usr/bin/env python3
"""Convert only locked EXP-021 scene metadata into the dataset camera format."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, yaml
from tttLRM.oracle.nuscenes_convert import load_tables, pose_matrix
from revisit3d.scripts.build_exp021_terminal_manifest import _sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/EXP-031_terminal_evaluation_v10.yaml"); ap.add_argument("--confirm-terminal-access",action="store_true")
    a=ap.parse_args()
    if not a.confirm_terminal_access: raise SystemExit("terminal metadata access requires confirmation")
    cp=Path(a.config); c=yaml.safe_load(cp.read_text()); out=Path(c["data"]["scene_root"]); result=Path(c["stage1"]["scene_conversion_result"])
    if result.exists() or out.exists(): raise RuntimeError("EXP-031 scene conversion output exists")
    auth=json.loads(Path(c["authorization"]).read_text()); lock=json.loads(Path(c["terminal_lock"]).read_text()); mp=Path(c["data"]["manifest"])
    if not (auth["registered_gate"]["passed"] and not auth["exp021_terminal_accessed"] and lock["terminal_test_locked"]
            and lock["manifest_sha256"]==_sha256(mp) and lock["directional_episodes"]==214): raise RuntimeError("terminal authorization failed")
    manifest=json.loads(mp.read_text()); wanted={r["source_scene"] for r in manifest}|{r["target_scene"] for r in manifest}
    root=c["data"]["nuscenes_root"]; version=c["data"]["nuscenes_version"]
    scenes,samples,sensors,calib,ego,sample_data=load_tables(root,version)
    cam={t:x for t,x in calib.items() if sensors[x["sensor_token"]]["channel"]=="CAM_FRONT"}; sample_scene={x["token"]:x["scene_token"] for x in samples}; by={}
    for rec in sample_data:
        if rec["calibrated_sensor_token"] in cam and rec["sample_token"] in sample_scene: by.setdefault(sample_scene[rec["sample_token"]],[]).append(rec)
    written=[]
    for scene in sorted((x for x in scenes if x["name"] in wanted),key=lambda x:x["name"]):
        records=sorted(by.get(scene["token"],[]),key=lambda x:x["timestamp"]); frames=[]
        for rec in records:
            cs=calib[rec["calibrated_sensor_token"]]; c2w=pose_matrix(ego[rec["ego_pose_token"]])@pose_matrix(cs); k=np.asarray(cs["camera_intrinsic"])
            frames.append({"w":rec["width"],"h":rec["height"],"fx":float(k[0,0]),"fy":float(k[1,1]),"cx":float(k[0,2]),"cy":float(k[1,2]),
                           "w2c":np.linalg.inv(c2w).tolist(),"file_path":str(Path(root)/rec["filename"])})
        d=out/scene["name"]; d.mkdir(parents=True,exist_ok=True); (d/"opencv_cameras.json").write_text(json.dumps({"scene_name":scene["name"],"frames":frames}))
        written.append({"scene":scene["name"],"frames":len(frames)})
    if {x["scene"] for x in written}!=wanted or min(x["frames"] for x in written)<200: raise RuntimeError("terminal scene conversion incomplete")
    payload={"experiment":"EXP-031","stage":"terminal_scene_metadata_conversion","protocol_revision":c["protocol_revision"],"config":str(cp),
             "manifest_sha256":_sha256(mp),"scenes":len(written),"minimum_frames":min(x["frames"] for x in written),"rows":written,
             "image_decoded":False,"lidar_decoded":False,"model_output_accessed":False,"terminal_access_started":True}
    result.parent.mkdir(parents=True,exist_ok=True); result.write_text(json.dumps(payload,indent=2)); print(json.dumps({k:payload[k] for k in ("scenes","minimum_frames","terminal_access_started")}))
if __name__=="__main__": main()
