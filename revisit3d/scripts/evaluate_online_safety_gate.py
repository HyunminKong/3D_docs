#!/usr/bin/env python3
"""Apply a parameter-free online loss gate to memory selection results."""
from __future__ import annotations
import argparse,json,torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--utility',required=True);p.add_argument('--out',required=True);a=p.parse_args();rows=[]
 for r in json.load(open(a.utility))['rows']:
  future=torch.tensor(r['candidate_utilities']);online=torch.tensor(r['candidate_current_objectives']);best=int(online.argmin());accept=bool(online[best] < r['current_context_objective']);selected=float(future[best] if accept else r['current']);rows.append({'accepted':accept,'selected_minus_current':selected-r['current'],'selected':selected,'current':r['current']})
 out={'summary':{'accept_rate':sum(x['accepted'] for x in rows)/len(rows),'mean_selected_minus_current':sum(x['selected_minus_current'] for x in rows)/len(rows),'harm_rate':sum(x['selected_minus_current']>0 for x in rows)/len(rows)},'rows':rows};json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(out['summary']))
if __name__=='__main__':main()
