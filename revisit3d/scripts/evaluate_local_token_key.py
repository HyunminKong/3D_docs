#!/usr/bin/env python3
"""Test training-free local-token set matching as a revisit context key."""
from __future__ import annotations
import argparse,json
import torch
from torch.nn import functional as F
from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.scripts.train_oracle_revisit import to_device
def desc(ex,seg):
 f=ex(seg['context']['rgb'])[0] # V,P,D
 # Four time slices × 64 spatially distributed patch tokens.
 f=f[torch.linspace(0,f.shape[0]-1,min(4,f.shape[0])).long()][:,::4].reshape(-1,f.shape[-1])
 return F.normalize(f,dim=-1)
def score(a,b):
 return float(.5*((a@b.T).max(1).values.mean()+(b@a.T).max(1).values.mean()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--split',default='val');p.add_argument('--out',required=True);a=p.parse_args();ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split=a.split,image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();items=[]
 for s in ds:
  A,B,P=(to_device(s[k],'cuda') for k in ('a','b','a_prime'));items.append((s['episode_id'],desc(ex,A),desc(ex,B),desc(ex,P)))
 rows=[]
 for i,(eid,A,B,P) in enumerate(items):
  scores=[score(A,x[3]) for x in items];rank=1+sum(x>scores[i] for x in scores);rows.append({'episode':eid,'positive':scores[i],'b':score(A,B),'rank':rank,'top1':rank==1})
 out={'rows':rows,'summary':{'top1':sum(x['top1'] for x in rows)/len(rows),'mean_rank':sum(x['rank'] for x in rows)/len(rows),'positive_minus_b':sum(x['positive']-x['b'] for x in rows)/len(rows)}};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
