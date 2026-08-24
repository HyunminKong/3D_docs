#!/usr/bin/env python3
"""Utility-based evaluation of learned local-token retrieval plus online reranking."""
from __future__ import annotations
import argparse,json,torch
from revisit3d.backbones import FrozenVGGTFeatures
from revisit3d.data import RevisitEpisodeDataset
from revisit3d.models.local_key import LocalTokenKey
from revisit3d.scripts.train_oracle_revisit import to_device

def token_key(ex,key,s):
 f=ex(s['context']['rgb'])[0];views=torch.linspace(0,f.shape[0]-1,4,device=f.device).long();return key(f[views,::4].reshape(-1,f.shape[-1]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--utility',required=True);p.add_argument('--key-checkpoint',default='revisit3d/checkpoints/local_token_key_dev.pt');p.add_argument('--k',type=int,default=5);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--vggt-checkpoint',default='FastVGGT/ckpt/model_tracker_fixed_e20.pt');p.add_argument('--out',required=True);a=p.parse_args()
 records=json.load(open(a.utility))['rows'];ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split='val',image_size=(224,224));ex=FrozenVGGTFeatures(a.vggt_checkpoint,repo_root='FastVGGT').cuda();key=LocalTokenKey(ex.feature_dim).cuda();key.load_state_dict(torch.load(a.key_checkpoint,map_location='cuda',weights_only=False)['key']);key.eval();mem=[];qry=[]
 with torch.no_grad():
  for x in ds:
   A,P=(to_device(x[t],'cuda') for t in ('a','a_prime'));mem.append(token_key(ex,key,A));qry.append(token_key(ex,key,P))
 rows=[]
 for q,r in zip(qry,records):
  scores=torch.tensor([float(key.score(q,m)) for m in mem]);cand=scores.topk(a.k).indices;future=torch.tensor(r['candidate_utilities']);online=torch.tensor(r['candidate_current_objectives']);oracle=int(future.argmin());chosen=int(cand[online[cand].argmin()]);rows.append({'oracle_in_shortlist':bool((cand==oracle).any()),'chosen_is_oracle':chosen==oracle,'selected_minus_current':float(future[chosen]-r['current']),'regret':float(future[chosen]-future[oracle])})
 out={'summary':{'oracle_shortlist_coverage':sum(x['oracle_in_shortlist'] for x in rows)/len(rows),'chosen_oracle_utility':sum(x['chosen_is_oracle'] for x in rows)/len(rows),'mean_selected_minus_current':sum(x['selected_minus_current'] for x in rows)/len(rows),'mean_regret':sum(x['regret'] for x in rows)/len(rows)},'rows':rows};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
