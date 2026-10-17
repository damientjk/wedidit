"""Training-only search for precision/recall at 20% workload. No val/test access."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime, timezone
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiment_capacity import ROOT, FEATURES, TARGET, build, digest, queue, validate_frame, write_json


def model_for(c,y):
    if c['family']=='hist_gradient_boosting':
        p=build({'family':'hist_gradient_boosting','leaves':c['leaves'],'balanced':False},y)
        p.named_steps['classifier'].set_params(max_iter=c['trees'],l2_regularization=c['reg'])
        return p
    p=build({'family':'xgboost','depth':c.get('depth',2),'trees':c.get('trees',200),'weight':c.get('weight','sqrt_ratio')},y)
    if c['family']=='forest':
        p.set_params(classifier=RandomForestClassifier(n_estimators=200,max_depth=12,
            min_samples_leaf=c['leaf'],max_features=.7,class_weight=c['class_weight'],
            random_state=42,n_jobs=1))
    else:
        p.named_steps['classifier'].set_params(min_child_weight=c.get('child',5),
            reg_lambda=c.get('reg',5),reg_alpha=c.get('alpha',0),subsample=c.get('subsample',1))
    return p


def main():
    start=time.monotonic()
    out=ROOT/'models/experiments'/('precision_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    protected={p:digest(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    path=ROOT/'data/processed/train.csv'
    tr=pd.read_csv(path,encoding='utf-8-sig');validate_frame(tr,'train','train')
    assert tr['split'].eq('train').all()
    x=tr[FEATURES];y=tr[TARGET].to_numpy();ids=tr.claim_id.to_numpy()
    configs=[{'id':'locked_xgb_control','family':'xgboost','depth':2,'trees':200,'weight':'sqrt_ratio'}]
    for depth,trees,child,reg,alpha in ((2,500,1,1,0),(3,400,1,5,0),(3,600,10,10,1),
                                       (4,400,5,10,1),(5,300,10,10,1),(3,300,20,20,2)):
        for weight in ('one','sqrt_ratio','ratio'):
            configs.append(dict(id=f'xgb_{len(configs):02}',family='xgboost',depth=depth,trees=trees,
                                child=child,reg=reg,alpha=alpha,weight=weight,subsample=.85))
    for leaves,trees,reg in ((7,400,10),(15,300,10),(15,400,20),(31,200,20)):
        configs.append(dict(id=f'hist_{len(configs):02}',family='hist_gradient_boosting',leaves=leaves,trees=trees,reg=reg))
    for leaf,weight in ((5,None),(15,None),(5,'balanced_subsample'),(15,'balanced_subsample')):
        configs.append(dict(id=f'forest_{len(configs):02}',family='forest',leaf=leaf,class_weight=weight))
    write_json(out/'protocol.json',{'selection':'mean fold precision at top20%, ties AP; ranking not score inflation',
        'features':FEATURES,'train_sha256':digest(path),'target':TARGET,'configs':configs,
        'search':'3-fold seed42 for all; top2 plus locked control confirmed with 3-fold seed137; choose best mean precision20 across both seeds, tie AP',
        'validation':'not opened; no claimed validation improvement','resources':'sequential; one native thread; RF200, XGB max600 trees'})
    records=[]
    def evaluate(c,seed):
        fold_rows=[]
        for fold,(fit_idx,hold_idx) in enumerate(StratifiedKFold(n_splits=3,shuffle=True,random_state=seed).split(x,y)):
            model=model_for(c,y[fit_idx]);model.fit(x.iloc[fit_idx],y[fit_idx])
            scores=model.predict_proba(x.iloc[hold_idx])[:,1]
            row={'id':c['id'],'seed':seed,'fold':fold,'ap':float(average_precision_score(y[hold_idx],scores))}
            for cap in (.1,.2,.3):
                q=queue(y[hold_idx],scores,ids[hold_idx],round(len(hold_idx)*cap))
                row[f'precision{int(cap*100)}']=q['precision'];row[f'recall{int(cap*100)}']=q['recall']
            records.append(row);fold_rows.append(row)
        print('COMPLETED',c['id'],'seed',seed,flush=True)
    for c in configs:evaluate(c,42)
    frame=pd.DataFrame(records)
    metrics=['precision10','recall10','precision20','recall20','precision30','recall30','ap']
    rank=frame.groupby('id')[metrics].mean().sort_values(['precision20','ap'],ascending=False)
    rank.to_csv(out/'search_ranking.csv')
    confirm=list(dict.fromkeys(rank.head(2).index.tolist()+['locked_xgb_control']))
    byid={c['id']:c for c in configs}
    write_json(out/'confirmation_selection.json',{'locked_before_second_seed':confirm})
    for name in confirm:evaluate(byid[name],137)
    frame=pd.DataFrame(records);frame.to_csv(out/'fold_metrics.csv',index=False)
    means=frame[frame.id.isin(confirm)].groupby('id')[metrics].mean().sort_values(['precision20','ap'],ascending=False)
    means.to_csv(out/'confirmed_comparison.csv')
    best=str(means.index[0]);pipeline=model_for(byid[best],y);pipeline.fit(x,y)
    joblib.dump(pipeline,out/'candidate_pipeline.joblib')
    check=joblib.load(out/'candidate_pipeline.joblib')
    np.testing.assert_allclose(pipeline.predict_proba(x.head(10)),check.predict_proba(x.head(10)))
    assert all(digest(p)==h for p,h in protected.items())
    write_json(out/'results.json',{'selected':best,'configuration':byid[best],
        'cv_comparison':means.reset_index().to_dict('records'),'elapsed_seconds':time.monotonic()-start,
        'validation_evaluated':False,'test_evaluated':False,'deployed':False,
        'protected_existing_artifacts_unchanged':True,
        'caveat':'Search/confirmation use the same training rows under different folds, not independent holdout; selected CV metrics can be optimistic. No new validation claim is made.'})
    print('OUTPUT',out,flush=True);print(means.to_string(),flush=True)
    print('SELECTED',best,byid[best],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
