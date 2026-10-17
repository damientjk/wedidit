"""Apply the team's revised policy to cached scores. No fitting or scoring.

The original frozen test protocol and outputs are preserved. The test-policy
comparison is post-evaluation analysis, not a new untouched holdout evaluation.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
OLD_LOW=.25596985
NEW_LOW=.1754
HIGH=.31254968

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(frame,score_column,label_column,low):
    p=frame[score_column].to_numpy();y=frame[label_column].to_numpy(dtype=int)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and set(y)=={0,1}
    tiers=np.where(p>=HIGH,'investigate',np.where(p>=low,'review','fast_track'))
    rows=[]
    for tier in ['fast_track','review','investigate']:
        mask=tiers==tier;n=int(mask.sum());fraud=int(y[mask].sum())
        rows.append(dict(tier=tier,claims=n,share=n/len(y),fraud=fraud,fraud_rate=fraud/n if n else None))
    flag=p>=low;tp=int(y[flag].sum());fp=int(flag.sum())-tp;fn=int(y[~flag].sum());tn=int((~flag).sum())-fn
    return dict(review_threshold=low,investigate_threshold=HIGH,claims=len(y),fraud=int(sum(y)),tiers=rows,
                fast_track_fraud_fraction=fn/sum(y),review_plus_investigate_claims=int(flag.sum()),
                review_plus_investigate_workload=float(flag.mean()),review_plus_investigate_recall=tp/sum(y),
                review_plus_investigate_precision=tp/int(flag.sum()),
                review_plus_investigate_f1=2*tp/(2*tp+fp+fn),confusion_matrix_review_plus_investigate=[[tn,fp],[fn,tp]])

def main():
    protected={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'docs/figures/test').rglob('*') if p.is_file()}
    model_path=ROOT/'models/experiments/capacity_20261017T042040511095Z/xgboost_09.joblib'
    model_hash=digest(model_path)
    assert model_hash=='bfa2cc3ca893eb9f0f2d066472cd6b7897155ce4d2270c5a849711fc8ffb76e9'
    manifest=json.loads((ROOT/'reports/explain_runtime/manifest.json').read_text())
    assert manifest['review_min_score']==NEW_LOW and manifest['investigate_min_score']==HIGH
    paths={'validation':ROOT/'docs/figures/validation_predictions.csv','test':ROOT/'docs/figures/test/test_predictions.csv'}
    results={}
    for split,path in paths.items():
        data=pd.read_csv(path);assert data.claim_id.is_unique
        score,label=('xgboost_09','actual_fraud') if split=='validation' else ('final_score','y_true')
        old=stats(data,score,label,OLD_LOW);new=stats(data,score,label,NEW_LOW)
        results[split]=dict(old=old,current=new,predictions_sha256=digest(path))
        assert old['tiers'][2]==new['tiers'][2]
        for policy,values in [('old',old),('current',new)]:
            assert sum(r['claims'] for r in values['tiers'])==values['claims']
            assert sum(r['fraud'] for r in values['tiers'])==values['fraud']
            pd.DataFrame(values['tiers']).to_csv(OUT/f'{split}_{policy}_tiers.csv',index=False)
        data['current_tier']=np.where(data[score]>=HIGH,'investigate',np.where(data[score]>=NEW_LOW,'review','fast_track'))
        data[['claim_id',label,score,'current_tier']].to_csv(OUT/f'{split}_current_routing.csv',index=False)
        # Keep the old and new distributions side by side; actual fraud labels are used only for reporting.
        fig,axes=plt.subplots(1,2,figsize=(10,5.6),sharey=True)
        for ax,values,title in zip(axes,[old,new],['Original review boundary 0.25596985','Current review boundary 0.1754']):
            fraud=np.array([r['fraud'] for r in values['tiers']]);total=np.array([r['claims'] for r in values['tiers']])
            ax.bar(['Fast-track','Review','Investigate'],total-fraud,label='Non-fraud',color='#6b9ac4')
            ax.bar(['Fast-track','Review','Investigate'],fraud,bottom=total-fraud,label='Fraud',color='#df7d48')
            for i,(n,f) in enumerate(zip(total,fraud)):ax.text(i,n+40,f'{n:,}\n({f} fraud)',ha='center',fontsize=10)
            ax.set_title(title,fontsize=11);ax.set_ylim(0,max(r['claims'] for r in old['tiers']+new['tiers'])*1.22)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Claims');axes[1].legend(loc='upper right',fontsize=9)
        fig.suptitle('Validation routing trade-off' if split=='validation' else 'Saved test routing reanalysis | policy revised after evaluation',fontsize=13)
        fig.text(.5,.02,f'n={len(data):,}; investigation boundary unchanged at {HIGH:.8f}; no retraining or rescoring',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUT/f'{split}_routing_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    assert results['validation']['old']['tiers'][0]['fraud']==77
    assert results['validation']['current']['tiers'][0]['fraud']==9
    assert results['test']['old']['tiers'][0]['fraud']==94
    assert results['test']['current']['tiers'][0]['fraud']==11
    assert protected=={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'docs/figures/test').rglob('*') if p.is_file()}
    assert digest(model_path)==model_hash
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),model_id='xgboost_09',model_sha256=model_hash,
                policy_source_commit='bc85518cab0bf1fe97c238adee588d1f7a7fcb5c',
                method='Apply fixed team-provided boundary to cached predictions; no model loading/fitting/rescoring or new raw test access.',
                interpretation='Routing policy revised after original test results were available; revised test routing is post-evaluation reanalysis, not an independent confirmatory test.',
                original_frozen_test_artifacts_unchanged=True,results=results)
    (OUT/'routing_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
