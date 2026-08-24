#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
import torch
from revisit3d.backbones import FrozenVGGTFeatures,FrozenVGGTGeometryTracker
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models import GeometryContextKey
from revisit3d.scripts.diagnose_track_ttt_signal import query_grid
from revisit3d.scripts.train_oracle_revisit import to_device
def enc(ex,tr,key,s,side):
 im=s['context']['rgb'];return key(ex(im),tr(im,query_grid(im.shape[-2],im.shape[-1],side,'cuda')))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--split',default='val');p.add_argument('--track-side',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args();ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split=a.split,image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();tr=FrozenVGGTGeometryTracker(a.vggt_checkpoint,repo_root='FastVGGT').cuda();key=GeometryContextKey(ex.feature_dim).cuda();key.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['key']);key.eval();items=[]
 with torch.no_grad():
  for s in ds:
   A,B,P=(to_device(s[k],'cuda') for k in ('a','b','a_prime'));items.append((s['episode_id'],enc(ex,tr,key,A,a.track_side),enc(ex,tr,key,B,a.track_side),enc(ex,tr,key,P,a.track_side)))
 rows=[]
 for i,(eid,ka,kb,kp) in enumerate(items):
  scores=torch.cat([ka@x[3].T for x in items],-1)[0];rank=int((scores>scores[i]).sum())+1;rows.append({'episode':eid,'positive':float(scores[i]),'b':float((ka@kb.T)[0,0]),'top1':rank==1,'rank':rank})
 summary={'top1':sum(x['top1'] for x in rows)/len(rows),'mean_rank':sum(x['rank'] for x in rows)/len(rows),'positive_minus_b':sum(x['positive']-x['b'] for x in rows)/len(rows)};json.dump({'rows':rows,'summary':summary},open(a.out,'w'),indent=2);print(json.dumps(summary))
if __name__=='__main__':main()
