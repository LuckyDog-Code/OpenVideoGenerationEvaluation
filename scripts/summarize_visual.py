"""Summarize three visual evaluation folders using paired, stratified prompts."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',action='append',required=True,help='NAME=EVALUATION_DIRECTORY (repeat)')
    ap.add_argument('--prompts',type=Path,default=Path(__file__).resolve().parents[1]/'data/prompts_30.jsonl')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    prompts=[json.loads(x) for x in a.prompts.read_text(encoding='utf8').splitlines() if x.strip()]
    dims=['subject_consistency','motion_smoothness','dynamic_degree','imaging_quality','aesthetic_quality','clip_x100']
    values={};view=None
    for entry in a.model:
        model,folder=entry.split('=',1)
        assert model not in values
        rows=[]
        for p in prompts:
            record=json.loads((Path(folder)/(p['prompt_id']+'_'+str(p['seed'])+'.json')).read_text())
            assert record['prompt_id']==p['prompt_id']
            actual=record['spec']['view']
            if view is None:view=actual
            assert actual==view,'Do not mix native and standardized views.'
            rows.append([record['scores'][d] for d in dims])
        values[model]=np.asarray(rows)
    strata=[[i for i,p in enumerate(prompts) if p['selection_group']==c] for c in sorted({p['selection_group'] for p in prompts})]
    rng=np.random.default_rng(20261118)
    draws=np.hstack([rng.choice(s,(4000,len(s))) for s in strata])
    contrasts={}
    for x,y in itertools.combinations(values,2):
        d=values[x]-values[y]
        contrasts[x+' minus '+y]=dict(mean=d.mean(0).tolist(),interval=np.quantile(d[draws].mean(1),[.025,.975],axis=0).tolist())
    result=dict(view=view,dimensions=dims,prompts=len(prompts),means={m:v.mean(0).tolist() for m,v in values.items()},paired=contrasts,
                bootstrap='4000 category-stratified paired prompt resamples; fixed generation seed; no seed variance')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':main()
