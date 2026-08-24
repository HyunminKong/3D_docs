#!/usr/bin/env python3
"""Does a current self-supervised score select a future-useful memory atom?"""
from __future__ import annotations
import argparse,json
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--utility',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 rows=[]
 for r in json.load(open(a.utility))['rows']:
  future=torch.tensor(r['candidate_utilities']);online=torch.tensor(r['candidate_current_objectives']);oracle=int(future.argmin());choice=int(online.argmin());top3=online.topk(3,largest=False).indices
  rows.append({'online_top1_is_oracle':choice==oracle,'oracle_in_online_top3':bool((top3==oracle).any()),
               'online_regret':float(future[choice]-future[oracle]),'current_regret':float(r['current']-future[oracle]),
               'online_selected_minus_current':float(future[choice]-r['current'])})
 out={'summary':{'online_top1_oracle_utility':sum(r['online_top1_is_oracle'] for r in rows)/len(rows),
                 'online_top3_oracle_coverage':sum(r['oracle_in_online_top3'] for r in rows)/len(rows),
                 'mean_online_regret':sum(r['online_regret'] for r in rows)/len(rows),
                 'mean_current_regret':sum(r['current_regret'] for r in rows)/len(rows),
                 'mean_online_selected_minus_current':sum(r['online_selected_minus_current'] for r in rows)/len(rows)},'rows':rows}
 json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
