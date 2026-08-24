#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,torch
from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models.local_key import LocalTokenKey
from revisit3d.scripts.train_oracle_revisit import to_device
def tok(ex,m,s):
 f=ex(s['context']['rgb'])[0];return m(f[torch.linspace(0,f.shape[0]-1,min(4,f.shape[0])).long()][:,::4].reshape(-1,f.shape[-1]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--out',required=True);a=p.parse_args();ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split='val',image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();m=LocalTokenKey(ex.feature_dim).cuda();m.load_state_dict(torch.load(a.checkpoint,map_location='cuda',weights_only=False)['key']);m.eval();mem=[];qry=[]
 with torch.no_grad():
  for s in ds:
   A,P=(to_device(s[k],'cuda') for k in ('a','a_prime'));mem.append(tok(ex,m,A));qry.append(tok(ex,m,P))
 rows=[]
 for i,q in enumerate(qry):
  scores=torch.tensor([float(m.score(q,k)) for k in mem]); order=scores.argsort(descending=True);rank=int((order==i).nonzero()[0])+1;rows.append({'rank':rank,'top1_score':float(scores[order[0]]),'margin':float(scores[order[0]]-scores[order[1]]),'correct_top1':rank==1})
 out={'rows':rows,'summary':{f'recall@{k}':sum(r['rank']<=k for r in rows)/len(rows) for k in (1,3,5,10)}};out['summary']['mean_rank']=sum(r['rank'] for r in rows)/len(rows);json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
