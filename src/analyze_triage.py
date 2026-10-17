"""Analyze locked candidate scores; never retrain or open test.csv."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models/experiments/capacity_20261017T042040511095Z"
OUT = ROOT / "reports/investigation_triage"
MODELS = ("xgboost_09", "hist_gradient_boosting_32")
BUDGETS = ((.05,.15),(.10,.20),(.10,.30),(.15,.35),(.20,.40))


def binary_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    precision = float(tp/(tp+fp)) if tp+fp else 0.
    return dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                precision=precision, recall=float(tp/(tp+fn)),
                f1=float(f1_score(y,pred,zero_division=0)), accuracy=float(accuracy_score(y,pred)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])


def main():
    original = json.loads((RUN/"results.json").read_text())
    for path, value in original["protected_model_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value
    for name, expected in (("train",original["protocol"]["training_sha256"]),
                           ("val",original["validation_sha256"])):
        assert hashlib.sha256((ROOT/f"data/processed/{name}.csv").read_bytes()).hexdigest()==expected
    frame = pd.read_csv(RUN/"validation_predictions.csv")
    y=frame.actual_fraud.to_numpy()
    n=len(y); positives=int(y.sum())
    rows=[]; goals=[]; frontiers=[]; classifications=[]
    for model in MODELS:
        scores=frame[model].to_numpy()
        order=np.lexsort((frame.claim_id.to_numpy(),-scores))
        cumulative=np.r_[0,np.cumsum(y[order])]
        ap=float(average_precision_score(y,scores)); auc=float(roc_auc_score(y,scores))
        classifications.append({"model":model,"metric_type":"classification at score >=0.5",
                                "threshold":.5,"ap":ap,"roc_auc":auc,**binary_metrics(y,scores>=.5)})
        # Minimum k for every attainable TP count. Plateaus are dominated.
        for tp in range(1,positives+1):
            k=int(np.flatnonzero(cumulative>=tp)[0])
            frontiers.append(dict(model=model,reviewed=k,workload=k/n,fraud_caught=tp,
                                  combined_coverage=tp/positives,false_positives=k-tp))
        for target in (.70,.80,.90,.95):
            required=int(np.ceil(target*positives))
            k=int(np.flatnonzero(cumulative>=required)[0])
            goals.append(dict(model=model,target=target,required_fraud=required,reviewed=k,
                              workload=k/n,actual_coverage=float(cumulative[k]/positives),
                              within_30pct_budget=k<=round(n*.30),within_60pct_budget=k<=round(n*.60)))
        for inv,review in BUDGETS:
            ki=round(n*inv); k=round(n*(inv+review)); kr=k-ki
            pred=np.zeros(n,dtype=bool);pred[order[:k]]=True
            metrics=binary_metrics(y,pred)
            ti=int(cumulative[ki]);tr=int(cumulative[k]-ti)
            earliest=int(np.flatnonzero(cumulative>=cumulative[k])[0])
            rows.append(dict(model=model,investigate_budget=inv,review_budget=review,
                investigate_claims=ki,review_claims=kr,fast_track_claims=n-k,
                investigate_fraud=ti,review_fraud=tr,fast_track_fraud=positives-ti-tr,
                investigate_false_positives=ki-ti,review_false_positives=kr-tr,
                investigate_only_recall=ti/positives,combined_coverage=metrics['recall'],
                human_escalations=k,workload=k/n,escalation_precision=metrics['precision'],
                average_precision=ap,roc_auc=auc,minimum_k_for_same_coverage=earliest,
                metric_type="capacity ranking; binary positive = escalated, not confirmed fraud",
                **metrics))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).drop(columns='confusion_matrix').to_csv(OUT/'policies.csv',index=False)
    pd.DataFrame(goals).to_csv(OUT/'coverage_goals.csv',index=False)
    pf=pd.DataFrame(frontiers)
    # Global frontier compares both locked models at each minimal workload.
    unique=pf.sort_values(['reviewed','fraud_caught'],ascending=[True,False]).drop_duplicates('reviewed')
    best=-1; keep=[]
    for row in unique.to_dict('records'):
        if row['fraud_caught']>best:
            keep.append(row);best=row['fraud_caught']
    pf.to_csv(OUT/'model_frontiers.csv',index=False)
    pd.DataFrame(keep).to_csv(OUT/'combined_frontier.csv',index=False)
    report=dict(source_run=str(RUN.relative_to(ROOT)),validation_rows=n,validation_fraud=positives,
                status="Exploratory policy analysis on previously-used validation; no tuning/retraining",
                classification_at_0_5=classifications,policies=rows,coverage_goals=goals,
                practical_budget_assumption="30% combined, 10% specialist +20% lightweight; not an approved staffing budget",
                operational_note="Exact top-k requires a defined batch and stable tie-break. All actual fraud counts are retrospective ground truth, never live confirmation.")
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows)[['model','investigate_budget','review_budget','investigate_claims',
        'review_claims','investigate_fraud','review_fraud','fast_track_fraud','workload',
        'precision','recall','f1','accuracy','balanced_accuracy','minimum_k_for_same_coverage']].to_string(index=False))
    print(pd.DataFrame(goals).to_string(index=False))
    print('CLASSIFICATION',json.dumps(classifications))


if __name__=='__main__':
    main()
