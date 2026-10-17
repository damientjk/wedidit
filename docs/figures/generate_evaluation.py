"""Reproduce validation-only evaluation from commit 48e14f9's locked artifacts.

No fitting, test-set reads, model replacement, API changes or deployment.
Run in Marcus's compatible environment: python docs/figures/generate_evaluation.py
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss)
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/figures'
CAP = ROOT / 'models/experiments/capacity_20261017T042040511095Z'
PROB = ROOT / 'models/experiments/probability_20261017T050105523504Z'
sys.path.insert(0, str(ROOT / 'src'))
import explain


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches='tight')
    plt.close()


def binary(y, scores, threshold):
    flag = scores >= threshold
    cm = confusion_matrix(y, flag, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())
    return dict(threshold=float(threshold), precision=float(precision_score(y, flag, zero_division=0)),
                recall=float(recall_score(y, flag)), f1=float(f1_score(y, flag)),
                false_positive_rate=fp/(tn+fp), flagged=int(sum(flag)),
                flagged_share=float(np.mean(flag)), confusion_matrix=cm.tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, pipeline, features, reference, encoder, categorical, owners = explain._load()
    prediction_bytes = (CAP / 'validation_predictions.csv').read_bytes()
    crlf_hash = hashlib.sha256(prediction_bytes.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
    expected_hash = manifest['validation_predictions_sha256']
    assert expected_hash in [digest(CAP / 'validation_predictions.csv'), crlf_hash]
    newline_note = 'Recorded Windows CRLF hash verified; Git checkout has LF line endings.' if digest(CAP / 'validation_predictions.csv') != expected_hash else 'Exact byte hash matched.'
    validation_path = ROOT / 'data/processed/val.csv'
    val = pd.read_csv(validation_path)
    saved = pd.read_csv(CAP / 'validation_predictions.csv')
    assert val.claim_id.is_unique and saved.claim_id.is_unique
    aligned = val.merge(saved, on='claim_id', how='inner', validate='one_to_one')
    assert len(aligned) == len(val) == len(saved) == 3084
    assert np.array_equal(aligned.FraudFound_P, aligned.actual_fraud)
    y = aligned.actual_fraud.to_numpy(dtype=int)
    assert set(y) == {0, 1} and sum(y) == 184
    scores = pipeline.predict_proba(aligned[features])[:, 1].astype(float)
    max_error = float(np.max(np.abs(scores-aligned.xgboost_09)))
    assert max_error < 1e-6, max_error
    aligned['xgboost_09'] = scores
    names = {'working_logistic':'Original LR baseline', 'logistic_02':'Tuned LR',
             'xgboost_09':'Locked tuned XGBoost', 'hist_gradient_boosting_32':'HistGradientBoosting reference'}
    comparison = []
    for key, name in names.items():
        s = aligned[key].to_numpy()
        assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
        comparison.append(dict(model=key, label=name, roc_auc=float(roc_auc_score(y,s)),
                               average_precision=float(average_precision_score(y,s))))
    reported = json.loads((CAP / 'results.json').read_text())
    for row in comparison:
        expected = next(item for item in reported['validation'] if item['model']==row['model'])
        assert abs(row['roc_auc']-expected['roc_auc']) < 1e-6
        assert abs(row['average_precision']-expected['average_precision']) < 1e-6
    pd.DataFrame(comparison).to_csv(OUT/'validation_model_comparison.csv',index=False)
    # ROC and PR plots compare ranking; the histogram model is a reference, not a new selection.
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    plt.figure(figsize=(8,6))
    for row in comparison:
        fpr,tpr,_=roc_curve(y,aligned[row['model']])
        plt.plot(fpr,tpr,label=f"{row['label']} (AUC {row['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'--',color='gray',label='Chance')
    plt.xlabel('False-positive rate');plt.ylabel('True-positive rate')
    plt.title('Validation ROC | 3,084 claims; 184 fraud labels');plt.legend(fontsize=9)
    save('roc_curve.png')
    plt.figure(figsize=(8,6))
    for row in comparison:
        precision,recall,_=precision_recall_curve(y,aligned[row['model']])
        plt.plot(recall,precision,label=f"{row['label']} (AP {row['average_precision']:.3f})")
    plt.axhline(y.mean(),ls='--',color='gray',label=f'Prevalence {y.mean():.2%}')
    plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Validation precision-recall | exploratory')
    plt.legend(fontsize=9);save('precision_recall_curve.png')
    # Reproduce the historical evaluation, not a later live routing policy.
    frozen=json.loads((OUT/'test/frozen_protocol.json').read_text())['thresholds']
    low=frozen['review']; high=frozen['investigate']
    operating={name:binary(y,scores,t) for name,t in [('investigate',high),('review_or_investigate',low)]}
    cm=np.array(operating['investigate']['confusion_matrix'])
    assert int(cm.sum())==len(y)
    plt.figure(figsize=(7.5,5.7));plt.imshow(cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(cm):
        plt.text(j,i,f'{value:,}',ha='center',va='center',fontsize=19,color='white' if value>cm.max()/2 else 'black')
    plt.xticks([0,1],['Below investigation\nthreshold','Investigate']);plt.yticks([0,1],['Non-fraud','Fraud'])
    plt.xlabel('Predicted routing');plt.ylabel('Actual label')
    plt.title(f'Validation confusion matrix | threshold {high:.8f}\nLocked tuned XGBoost; n={len(y):,}')
    save('confusion_matrix.png')
    tiers=np.where(scores>=high,'investigate',np.where(scores>=low,'review','fast_track'))
    tier_rows=[]
    for tier in ['fast_track','review','investigate']:
        mask=tiers==tier;count=int(mask.sum());fraud=int(y[mask].sum())
        tier_rows.append(dict(tier=tier,count=count,share=count/len(y),fraud=fraud,fraud_rate=fraud/count))
    assert sum(r['count'] for r in tier_rows)==len(y) and sum(r['fraud'] for r in tier_rows)==int(sum(y))
    pd.DataFrame(tier_rows).to_csv(OUT/'validation_tiers.csv',index=False)
    plt.figure(figsize=(7.5,5));labels=['Low / fast-track','Medium / review','High / investigate']
    fraud=np.array([r['fraud'] for r in tier_rows]);non=np.array([r['count']-r['fraud'] for r in tier_rows])
    plt.bar(labels,non,label='Non-fraud',color='#6b9ac4');plt.bar(labels,fraud,bottom=non,label='Fraud',color='#df7d48')
    plt.ylabel('Validation claims');plt.title('Provisional score bands | not automatic decisions');plt.legend()
    save('tier_distribution.png')
    capacity=[]
    for key,name in names.items():
        rank=aligned.sort_values([key,'claim_id'],ascending=[False,True],kind='mergesort')
        for fraction in [.1,.2,.3]:
            k=round(len(y)*fraction);caught=int(rank.actual_fraud.head(k).sum())
            expected=next(r for r in reported['capacity_metrics'] if r['model']==key and abs(r['capacity']-fraction)<1e-10)
            assert caught==expected['fraud_caught']
            capacity.append(dict(model=key,capacity=fraction,reviewed=k,caught=caught,precision=caught/k,recall=caught/sum(y)))
    pd.DataFrame(capacity).to_csv(OUT/'validation_capacity.csv',index=False)
    # Aggregate exact native TreeSHAP columns to the 19 original input fields.
    encoded=pipeline.named_steps['preprocessor'].transform(aligned[features])
    booster=pipeline.named_steps['classifier'].get_booster();dm=xgb.DMatrix(encoded,nthread=1)
    shap=booster.predict(dm,pred_contribs=True,approx_contribs=False)
    aggregated=np.column_stack([shap[:,:-1][:,np.array(owners)==f].sum(axis=1) for f in features])
    margin=booster.predict(dm,output_margin=True)
    additivity_error=float(np.max(np.abs(aggregated.sum(axis=1)+shap[:,-1]-margin)))
    assert additivity_error<2e-5
    assert np.max(np.abs(1/(1+np.exp(-margin))-scores))<1e-6
    importance=pd.DataFrame({'feature':features,'label':[explain.LABELS[f] for f in features],
                             'mean_absolute_log_odds':np.abs(aggregated).mean(axis=0)})
    importance=importance.sort_values('mean_absolute_log_odds',ascending=False)
    importance.to_csv(OUT/'feature_contributions.csv',index=False)
    display=importance.head(12).iloc[::-1]
    plt.figure(figsize=(9,6));plt.barh(display.label,display.mean_absolute_log_odds,color='#2f7f88')
    plt.xlabel('Mean absolute native TreeSHAP contribution (log-odds)')
    plt.title('Top 12 model features | all 3,084 validation claims\nLocked tuned XGBoost; associations, not causes')
    save('feature_contributions.png')
    probs=pd.read_csv(PROB/'validation_predictions.csv')
    assert probs.claim_id.is_unique and set(probs.claim_id)==set(aligned.claim_id)
    probs=aligned[['claim_id','actual_fraud']].merge(probs,on='claim_id',suffixes=('_source',''),validate='one_to_one')
    assert np.array_equal(probs.actual_fraud_source,probs.actual_fraud)
    probability_rows=[]
    for key in ['xgboost_raw','xgboost_isotonic','logistic_sigmoid','blend_sigmoid']:
        p=probs[key].to_numpy();probability_rows.append(dict(model=key,log_loss=float(log_loss(y,p)),
            brier=float(brier_score_loss(y,p)),roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p))))
    aligned[['claim_id','actual_fraud','working_logistic','logistic_02','xgboost_09']].assign(tier=tiers).to_csv(OUT/'validation_predictions.csv',index=False)
    result=dict(commit='48e14f9eb3aae344bb14c8f66ef05893276ad2a8',split='validation',test_evaluated=False,
        claims=len(y),fraud=int(sum(y)),comparison=comparison,thresholds=dict(review=low,investigate=high),
        operating=operating,tiers=tier_rows,capacity=capacity,probability_experiment=probability_rows,
        max_saved_prediction_difference=max_error,max_contribution_additivity_error=additivity_error,
        model_sha256=digest(ROOT/manifest['model_path']),validation_sha256=digest(validation_path),
        saved_predictions_sha256=digest(CAP/'validation_predictions.csv'),recorded_prediction_sha256=expected_hash,prediction_hash_note=newline_note,
        environment=dict(platform=platform.platform(),python=platform.python_version(),
            versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','matplotlib']}))
    (OUT/'evaluation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['claims','fraud','operating','tiers','max_saved_prediction_difference','max_contribution_additivity_error']},indent=2))

if __name__=='__main__':main()
