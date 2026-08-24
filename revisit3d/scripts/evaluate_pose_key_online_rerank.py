#!/usr/bin/env python3
"""Two-stage oracle map shortlist followed by an online TTT-utility reranker."""
from __future__ import annotations
import argparse,json,torch
from revisit3d.data import RevisitEpisodeDataset

def centres(s): return torch.linalg.inv(s['context']['w2c'])[:,:3,3]
def main():
 p=argparse.ArgumentParser();p.add_argument('--utility',required=True);p.add_argument('--k',type=int,default=3);p.add_argument('--manifest',default='revisit3d/manifests/nuscenes_revisit_dev.json');p.add_argument('--scene-root',default='tttLRM/data_example/nuscenes_2x2');p.add_argument('--out',required=True);a=p.parse_args()
 records=json.load(open(a.utility))['rows'];ds=RevisitEpisodeDataset(a.manifest,a.scene_root,split='val',image_size=(224,224));mem=[centres(x['a']) for x in ds];qry=[centres(x['a_prime']) for x in ds];rows=[]
 for q,r in zip(qry,records):
  key=torch.tensor([torch.cdist(q,m).min() for m in mem]);cand=key.topk(a.k,largest=False).indices;future=torch.tensor(r['candidate_utilities']);online=torch.tensor(r['candidate_current_objectives']);chosen=int(cand[online[cand].argmin()]);oracle=int(future.argmin());rows.append({'oracle_in_shortlist':bool((cand==oracle).any()),'chosen_is_oracle':chosen==oracle,'selected_minus_current':float(future[chosen]-r['current']),'regret':float(future[chosen]-future[oracle])})
 out={'summary':{'oracle_shortlist_coverage':sum(x['oracle_in_shortlist'] for x in rows)/len(rows),'chosen_oracle_utility':sum(x['chosen_is_oracle'] for x in rows)/len(rows),'mean_selected_minus_current':sum(x['selected_minus_current'] for x in rows)/len(rows),'mean_regret':sum(x['regret'] for x in rows)/len(rows)},'rows':rows};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
