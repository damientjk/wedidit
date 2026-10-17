"""Select calibrated LR/XGBoost probability models using training-only nested CV.

Run: python src/train_probability.py
All outputs go to a new models/experiments/probability_* directory. No UI work.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

from datetime import datetime, timezone
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from train import ROOT, FEATURES, TARGET, validate_frame
from train_xgboost import environment, sha256

SOURCE=ROOT/'models/experiments/capacity_20261017T042040511095Z'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def probability_metrics(y,p):
    pred=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'log_loss':float(log_loss(y,p,labels=[0,1])),
        'brier_score':float(brier_score_loss(y,p)),
        'roc_auc':float(roc_auc_score(y,p)), 'average_precision':float(average_precision_score(y,p)),
        'mean_probability':float(np.mean(p)), 'observed_fraud_rate':float(np.mean(y)),
        'actual_fraud_mean_probability':float(np.mean(p[y==1])),
        'actual_fraud_median_probability':float(np.median(p[y==1])),
        'actual_legitimate_mean_probability':float(np.mean(p[y==0])),
        'actual_legitimate_median_probability':float(np.median(p[y==0])),
        'at_0_5':{'accuracy':float(accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred)), 'f1':float(f1_score(y,pred,zero_division=0)),
            'specificity':float(tn/(tn+fp)),
            'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}}


def make_calibrated(base,method,seed):
    return CalibratedClassifierCV(estimator=clone(base), method=method,
        cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed),
        ensemble=False,n_jobs=1)


def reliability(y,p,model,kind):
    if kind=='quantile':
        groups=pd.qcut(p,q=10,duplicates='drop')
    else:
        groups=pd.cut(p,bins=[0,.01,.02,.05,.10,.15,.20,.30,.50,1],include_lowest=True)
    frame=pd.DataFrame({'bin':groups,'probability':p,'actual':y})
    result=[]
    for label,g in frame.groupby('bin',observed=True):
        result.append({'model':model,'binning':kind,'bin':str(label),'claims':len(g),
            'fraud':int(g.actual.sum()),'mean_probability':float(g.probability.mean()),
            'observed_fraud_rate':float(g.actual.mean())})
    return result


def main():
    started=time.monotonic()
    protected={p:sha256(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    out=ROOT/'models/experiments'/('probability_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    train_path=ROOT/'data/processed/train.csv'
    train=pd.read_csv(train_path,encoding='utf-8-sig')
    validate_frame(train,'train','train')
    assert train['split'].eq('train').all()
    x=train[FEATURES];y=train[TARGET].to_numpy()
    lr=clone(joblib.load(SOURCE/'logistic_02.joblib'))
    xgb=clone(joblib.load(SOURCE/'xgboost_09.joblib'))
    bases={'logistic':lr,'xgboost':xgb,
           'blend':VotingClassifier(estimators=[('logistic',clone(lr)),('xgboost',clone(xgb))],voting='soft',weights=[1,1],n_jobs=1)}
    protocol={'target':TARGET,'features':FEATURES,'training_sha256':sha256(train_path),
        'models':'Fixed previously training-CV-selected LR C=.1 unweighted; tuned XGBoost depth2/200/sqrt weighting; equal-weight soft-voting combination',
        'calibrations':['sigmoid','isotonic'],
        'selection':'lowest mean outer-fold log loss among six calibrated candidates; tie lower Brier score then name. Raw models and training-prevalence baseline are controls, not calibrated candidates.',
        'outer_cv':'3-fold StratifiedKFold(shuffle=True,random_state=2026)',
        'inner_cv':'3-fold shuffled, seed=100+outer_fold; ensemble=False uses inner OOF calibration predictions, then refits base on all outer-training rows',
        'weights':'calibrator uses natural unweighted labels; no class rebalancing or oversampling of calibration data',
        'final_fit':'inner OOF calibration across all training, seed42; full-training base refit; validation untouched during selection or fitting',
        'resources':'one thread, sequential, fixed base hyperparameters; no new base-model tuning',
        'limitations':'Previously chosen base hyperparameters used these training rows; nested calibration evaluation is not a fresh unbiased model-selection holdout. Validation is previously-used and exploratory.',
        'environment':environment()}
    write_json(out/'protocol.json',protocol)
    records=[]
    oof=pd.DataFrame({'claim_id':train.claim_id,'actual_fraud':y})
    for family in bases:
        for method in ('raw','sigmoid','isotonic'):
            oof[f'{family}_{method}']=np.nan
    oof['prevalence_control']=np.nan
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=2026).split(x,y))
    for fold,(fit_idx,hold_idx) in enumerate(folds):
        constant=np.full(len(hold_idx),float(y[fit_idx].mean()))
        oof.loc[hold_idx,'prevalence_control']=constant
        records.append({'model':'prevalence_control','fold':fold,**probability_metrics(y[hold_idx],constant)})
        for family,base in bases.items():
            for method in ('raw','sigmoid','isotonic'):
                name=f'{family}_{method}'
                model=clone(base) if method=='raw' else make_calibrated(base,method,100+fold)
                model.fit(x.iloc[fit_idx],y[fit_idx])
                p=model.predict_proba(x.iloc[hold_idx])[:,list(model.classes_).index(1)]
                oof.loc[hold_idx,name]=p
                records.append({'model':name,'fold':fold,**probability_metrics(y[hold_idx],p)})
                print('CV',fold,name,flush=True)
    folds_table=pd.DataFrame([{k:v for k,v in row.items() if k!='at_0_5'} for row in records])
    folds_table.to_csv(out/'cv_folds.csv',index=False)
    oof.to_csv(out/'training_oof_predictions.csv',index=False)
    rank=folds_table.groupby('model',as_index=False)[['log_loss','brier_score','roc_auc','average_precision']].mean().sort_values(['log_loss','brier_score','model'])
    rank.to_csv(out/'cv_comparison.csv',index=False)
    calibrated=rank[rank.model.str.endswith(('_sigmoid','_isotonic'))]
    selected=str(calibrated.iloc[0].model)
    per_family={family:str(calibrated[calibrated.model.str.startswith(family+'_')].iloc[0].model) for family in bases}
    write_json(out/'selection.json',{'selected':selected,'per_family':per_family,
        'locked_before_validation':True,'cv_scores':calibrated.to_dict('records')})
    print('LOCKED_SELECTION',selected,flush=True)
    fitted={}
    for family,base in bases.items():
        raw=clone(base).fit(x,y)
        fitted[f'{family}_raw']=raw
        name=per_family[family];method=name.rsplit('_',1)[1]
        fitted[name]=make_calibrated(base,method,42).fit(x,y)
        joblib.dump(fitted[name],out/f'{name}.joblib')
    joblib.dump(fitted[selected],out/'calibrated_probability.joblib')
    # Open frozen validation only after locking choices and fitting all finalists.
    val_path=ROOT/'data/processed/val.csv'
    val=pd.read_csv(val_path,encoding='utf-8-sig')
    validate_frame(val,'validation','val');assert val['split'].eq('val').all()
    assert set(val.claim_id).isdisjoint(set(train.claim_id))
    vx=val[FEATURES];vy=val[TARGET].to_numpy()
    predictions=pd.DataFrame({'claim_id':val.claim_id,'actual_fraud':vy})
    rows=[]; bins=[]; validation={}
    for name,model in fitted.items():
        p=model.predict_proba(vx)[:,list(model.classes_).index(1)]
        predictions[name]=p
        result=probability_metrics(vy,p)
        validation[name]=result
        rows.append({'model':name,'selected':name==selected,**{k:v for k,v in result.items() if k!='at_0_5'}})
        for kind in ('quantile','fixed'):
            bins.extend(reliability(vy,p,name,kind))
    const=np.full(len(vy),float(y.mean()))
    validation['prevalence_control']=probability_metrics(vy,const)
    rows.append({'model':'prevalence_control','selected':False,**{k:v for k,v in validation['prevalence_control'].items() if k!='at_0_5'}})
    table=pd.DataFrame(rows);table.to_csv(out/'validation_comparison.csv',index=False)
    predictions.to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    loaded=joblib.load(out/'calibrated_probability.joblib')
    np.testing.assert_allclose(loaded.predict_proba(vx),fitted[selected].predict_proba(vx))
    probe=vx.head(2).copy();probe.loc[probe.index[0],'Make']='UNSEEN_BRAND'
    probe.loc[probe.index[0],'Deductible']=np.nan;probe.loc[probe.index[0],'BasePolicy']=np.nan
    probabilities=loaded.predict_proba(probe)
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    assert all(sha256(p)==value for p,value in protected.items())
    final={'protocol':protocol,'selected':selected,'per_family':per_family,
        'train_rows':len(train),'train_fraud':int(y.sum()),'validation_rows':len(val),'validation_fraud':int(vy.sum()),
        'validation_sha256':sha256(val_path),'validation':validation,
        'verification':{'reload_predictions_match':True,'missing_unknown_probability_checks':True,
                        'protected_artifacts_unchanged':True,'test_access':False},
        'elapsed_seconds':time.monotonic()-started,
        'protected_sha256':{str(p.relative_to(ROOT)):value for p,value in protected.items()}}
    write_json(out/'metrics.json',final)
    write_json(out/'manifest.json',{'model_id':selected,'model_file':'calibrated_probability.joblib',
        'model_sha256':sha256(out/'calibrated_probability.joblib'),'features':FEATURES,
        'target':TARGET,'positive_label':1,'classes':loaded.classes_.tolist(),
        'output':'predict_proba(DataFrame[features])[:, index of class 1] = estimated fraud probability',
        'environment':environment(),'status':'candidate, not deployed; validation exploratory; population drift can invalidate calibration'})
    # Scientific chart, kept beside the separate model artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    selected_family=selected.rsplit('_',1)[0]
    for name in (selected_family+'_raw',selected):
        subset=[b for b in bins if b['model']==name and b['binning']=='quantile']
        axes[0].plot([b['mean_probability'] for b in subset],[b['observed_fraud_rate'] for b in subset],marker='o',label=name)
        axes[1].hist(predictions[name],bins=np.linspace(0,1,41),alpha=.5,label=name)
    axes[0].plot([0,1],[0,1],'k--',label='Perfect calibration')
    axes[0].set(xlabel='Mean estimated probability',ylabel='Observed fraud frequency',title='Validation reliability (quantile bins)',xlim=(0,.4),ylim=(0,.4))
    axes[1].set(xlabel='Estimated fraud probability / raw score',ylabel='Claims',title='Validation output distribution')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=160);plt.close(fig)
    print('OUTPUT',out,flush=True)
    print('CV',rank.to_string(index=False),flush=True)
    print('VALIDATION',table[['model','selected','log_loss','brier_score','roc_auc','average_precision','mean_probability']].to_string(index=False),flush=True)
    print('SELECTED_DETAILS',json.dumps(validation[selected],indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
