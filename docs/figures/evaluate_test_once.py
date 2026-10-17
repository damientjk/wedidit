"""Locked test evaluation, authorised by the team after model confirmation.

First invocation saves predictions once; --reuse regenerates summaries/figures
from those saved outputs without loading models or re-reading the test set.
No fitting, tuning, calibration comparison or deployment.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
from statistics import NormalDist
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/figures/test'
MODEL='models/experiments/capacity_20261017T042040511095Z/xgboost_09.joblib'
EXPECTED='bfa2cc3ca893eb9f0f2d066472cd6b7897155ce4d2270c5a849711fc8ffb76e9'
LOW=.25596985
HIGH=.31254968

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def wilson(success,total):
    if not total:return None
    z=NormalDist().inv_cdf(.975);p=success/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(center-half),float(center+half)]

def measure(y,p,t):
    flag=p>=t;cm=confusion_matrix(y,flag,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    return dict(threshold=float(t),precision=float(precision_score(y,flag,zero_division=0)),
                recall=float(recall_score(y,flag)),f1=float(f1_score(y,flag)),
                false_positive_rate=fp/(tn+fp),flagged=int(flag.sum()),workload=float(flag.mean()),
                confusion_matrix=cm.tolist(),recall_wilson_95=wilson(tp,tp+fn),
                precision_wilson_95=wilson(tp,tp+fp))

def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close()

def first_evaluation():
    import joblib
    import xgboost as xgb
    sys.path.insert(0,str(ROOT/'src'))
    import explain
    locked=json.loads((ROOT/'reports/explain_runtime/manifest.json').read_text())
    assert locked['model_path']==MODEL and locked['model_sha256']==EXPECTED
    assert locked['review_min_score']==LOW and locked['investigate_min_score']==HIGH
    assert sha(ROOT/MODEL)==EXPECTED
    protected={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    protocol=dict(source_commit='48e14f9eb3aae344bb14c8f66ef05893276ad2a8',
        authorisation='The team confirmed locked tuned XGBoost as final and authorised the once-only test evaluation.',
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),model=MODEL,model_sha256=EXPECTED,
        original_baseline='models/baseline.joblib',baseline_sha256=sha(ROOT/'models/baseline.joblib'),
        thresholds=dict(review=LOW,investigate=HIGH,baseline_binary=.5),
        capacity_fractions=[.1,.2,.3],ties='claim_id ascending',
        plan='Evaluate locked XGBoost and existing original logistic baseline. No other candidates, fitting or threshold selection. Save predictions before reporting.',
        environment=dict(platform=platform.platform(),python=platform.python_version(),
            versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','matplotlib']}))
    protocol_path=OUT/'frozen_protocol.json'
    if protocol_path.exists():raise RuntimeError('Protocol already exists; inspect prior attempt before another first evaluation.')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    # Protocol is frozen before test bytes or labels are opened.
    test_path=ROOT/'data/processed/test.csv';test_bytes=test_path.read_bytes()
    test=pd.read_csv(io.BytesIO(test_bytes))
    assert test.claim_id.is_unique and set(test.FraudFound_P.unique())=={0,1}
    assert set(test.split)=={'test'}
    for split in ['train','val']:
        other=set(pd.read_csv(ROOT/f'data/processed/{split}.csv',usecols=['claim_id']).claim_id)
        assert not set(test.claim_id)&other
    manifest,pipeline,features,reference,encoder,categorical,owners=explain._load()
    baseline=joblib.load(ROOT/'models/baseline.joblib')
    assert list(baseline.feature_names_in_)==features
    frame=test[features]
    xgb_score=pipeline.predict_proba(frame)[:,list(pipeline.classes_).index(1)].astype(float)
    baseline_score=baseline.predict_proba(frame)[:,list(baseline.classes_).index(1)].astype(float)
    assert np.isfinite(xgb_score).all() and np.isfinite(baseline_score).all()
    tiers=np.where(xgb_score>=HIGH,'investigate',np.where(xgb_score>=LOW,'review','fast_track'))
    preds=pd.DataFrame(dict(claim_id=test.claim_id.to_numpy(),y_true=test.FraudFound_P.to_numpy(),
                            lr_score=baseline_score,xgb_score=xgb_score,final_score=xgb_score,tier=tiers))
    preds.to_csv(OUT/'test_predictions.csv',index=False)
    # Native explanations for the frozen raw model, aggregated over original inputs.
    encoded=pipeline.named_steps['preprocessor'].transform(frame)
    booster=pipeline.named_steps['classifier'].get_booster();dm=xgb.DMatrix(encoded,nthread=1)
    raw=booster.predict(dm,pred_contribs=True,approx_contribs=False)
    contributions=np.column_stack([raw[:,:-1][:,np.array(owners)==f].sum(axis=1) for f in features])
    margin=booster.predict(dm,output_margin=True)
    error=float(np.max(np.abs(contributions.sum(axis=1)+raw[:,-1]-margin)))
    assert error<2e-5
    assert np.max(np.abs(1/(1+np.exp(-margin))-xgb_score))<1e-6
    importance=pd.DataFrame(dict(feature=features,label=[explain.LABELS[f] for f in features],
        mean_absolute_log_odds=np.abs(contributions).mean(axis=0))).sort_values('mean_absolute_log_odds',ascending=False)
    importance.to_csv(OUT/'feature_contributions.csv',index=False)
    assert protected=={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    audit=dict(test_sha256=hashlib.sha256(test_bytes).hexdigest(),prediction_sha256=sha(OUT/'test_predictions.csv'),
               model_artifacts_unchanged=True,protected_model_files=len(protected),
               no_split_id_overlap=True,max_contribution_additivity_error=error)
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2))

def report_saved():
    audit=json.loads((OUT/'verification.json').read_text())
    assert sha(OUT/'test_predictions.csv')==audit['prediction_sha256']
    preds=pd.read_csv(OUT/'test_predictions.csv');y=preds.y_true.to_numpy();p=preds.final_score.to_numpy()
    assert preds.claim_id.is_unique and np.array_equal(preds.xgb_score,p)
    assert all(np.isfinite(preds[c]).all() and preds[c].between(0,1).all() for c in ['lr_score','xgb_score','final_score'])
    expected=np.where(p>=HIGH,'investigate',np.where(p>=LOW,'review','fast_track'))
    assert np.array_equal(preds.tier,expected)
    comparisons=[]
    for col,name in [('lr_score','Original LR baseline'),('xgb_score','Locked tuned XGBoost')]:
        s=preds[col].to_numpy();comparisons.append(dict(model=col,label=name,roc_auc=float(roc_auc_score(y,s)),
            average_precision=float(average_precision_score(y,s))))
    pd.DataFrame(comparisons).to_csv(OUT/'model_comparison.csv',index=False)
    operating=dict(investigate=measure(y,p,HIGH),review_or_investigate=measure(y,p,LOW),
                   original_baseline_at_0_5=measure(y,preds.lr_score.to_numpy(),.5))
    capacity=[]
    for col,name in [('lr_score','Original LR baseline'),('xgb_score','Locked tuned XGBoost')]:
        ranked=preds.sort_values([col,'claim_id'],ascending=[False,True],kind='mergesort')
        for fraction in [.1,.2,.3]:
            k=round(len(y)*fraction);caught=int(ranked.y_true.head(k).sum())
            capacity.append(dict(model=name,fraction=fraction,reviewed=k,caught=caught,precision=caught/k,
                recall=caught/sum(y),recall_wilson_95=wilson(caught,int(sum(y)))))
    pd.DataFrame(capacity).to_csv(OUT/'capacity.csv',index=False)
    tiers=[]
    for tier in ['fast_track','review','investigate']:
        mask=preds.tier==tier;n=int(mask.sum());fraud=int(y[mask].sum())
        tiers.append(dict(tier=tier,count=n,share=n/len(y),fraud=fraud,fraud_rate=fraud/n if n else None))
    pd.DataFrame(tiers).to_csv(OUT/'tiers.csv',index=False)
    assert sum(t['count'] for t in tiers)==len(y) and sum(t['fraud'] for t in tiers)==sum(y)
    for op in operating.values():assert sum(map(sum,op['confusion_matrix']))==len(y)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    plt.figure(figsize=(8,6))
    for r in comparisons:
        fpr,tpr,_=roc_curve(y,preds[r['model']]);plt.plot(fpr,tpr,label=f"{r['label']} (AUC {r['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'--',color='gray',label='Chance');plt.xlabel('False-positive rate');plt.ylabel('True-positive rate')
    plt.title(f'Final test ROC | {len(y):,} claims; {sum(y)} fraud labels');plt.legend();save('roc_curve.png')
    plt.figure(figsize=(8,6))
    for r in comparisons:
        precision,recall,_=precision_recall_curve(y,preds[r['model']]);plt.plot(recall,precision,label=f"{r['label']} (AP {r['average_precision']:.3f})")
    plt.axhline(y.mean(),ls='--',color='gray',label=f'Prevalence {y.mean():.2%}')
    plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Final test precision-recall | frozen models');plt.legend();save('precision_recall_curve.png')
    cm=np.array(operating['investigate']['confusion_matrix']);plt.figure(figsize=(7.5,5.7));plt.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):plt.text(j,i,f'{v:,}',ha='center',va='center',fontsize=19,color='white' if v>cm.max()/2 else 'black')
    plt.xticks([0,1],['Below investigation\nthreshold','Investigate']);plt.yticks([0,1],['Non-fraud','Fraud'])
    plt.xlabel('Predicted routing');plt.ylabel('Actual label');plt.title(f'Final test confusion matrix | threshold {HIGH:.8f}\nLocked tuned XGBoost; n={len(y):,}');save('confusion_matrix.png')
    importance=pd.read_csv(OUT/'feature_contributions.csv').head(12).iloc[::-1]
    plt.figure(figsize=(9,6));plt.barh(importance.label,importance.mean_absolute_log_odds,color='#2f7f88')
    plt.xlabel('Mean absolute native TreeSHAP contribution (log-odds)');plt.title(f'Top 12 model features | all {len(y):,} test claims\nFrozen XGBoost; associations, not causes');save('feature_contributions.png')
    plt.figure(figsize=(7.5,5));fraud=np.array([r['fraud'] for r in tiers]);non=np.array([r['count']-r['fraud'] for r in tiers])
    labels=['Low / fast-track','Medium / review','High / investigate']
    plt.bar(labels,non,label='Non-fraud',color='#6b9ac4');plt.bar(labels,fraud,bottom=non,label='Fraud',color='#df7d48')
    plt.ylabel('Test claims');plt.title('Final test routing | unchanged validation-derived bands');plt.legend();save('tier_distribution.png')
    result=dict(split='final_test',model='xgboost_09',claims=len(y),fraud=int(sum(y)),fraud_rate=float(y.mean()),
        comparisons=comparisons,operating=operating,capacity=capacity,tiers=tiers,
        thresholds=dict(review=LOW,investigate=HIGH),verification=audit,
        protocol=json.loads((OUT/'frozen_protocol.json').read_text()))
    (OUT/'evaluation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['claims','fraud','comparisons','operating','capacity','tiers']},indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.reuse:
        if (OUT/'test_predictions.csv').exists():raise RuntimeError('Test predictions already exist. Use --reuse, not a fresh evaluation.')
        first_evaluation()
    report_saved()

if __name__=='__main__':main()
