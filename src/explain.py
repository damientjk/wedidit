"""Locked XGBoost inference. No training, claim-data reads or deployment.

score(claim, exclude=None) accepts the team's flat claim payload and returns
risk_score, tier, contributions, excluded plus explanation/safety metadata.
"""
from functools import lru_cache
import hashlib
import json
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "explain_runtime" / "manifest.json"
INTEGER_CATEGORIES = {"WeekOfMonth", "WeekOfMonthClaimed", "DriverRating"}
LABELS = {
    "Deductible":"Policy deductible", "Month":"Accident month",
    "WeekOfMonth":"Accident week of month", "DayOfWeek":"Accident weekday",
    "MonthClaimed":"Claim reporting month", "WeekOfMonthClaimed":"Claim reporting week of month",
    "DayOfWeekClaimed":"Claim reporting weekday", "Make":"Vehicle make",
    "AccidentArea":"Accident area", "VehicleCategory":"Vehicle category",
    "BasePolicy":"Policy coverage", "AgentType":"Agent type", "VehiclePrice":"Vehicle price band",
    "DriverRating":"Driver rating", "Days_Policy_Accident":"Policy-to-accident duration band",
    "Days_Policy_Claim":"Policy-to-claim duration band", "PastNumberOfClaims":"Prior claims band",
    "AgeOfVehicle":"Vehicle age band", "NumberOfCars":"Number of cars band",
}
RECOMMENDATIONS = {
    "investigate": "High priority: refer for specialist investigation. This risk flag is not a finding of fraud.",
    "review": "Medium priority: request lightweight human verification through standard processing. This risk flag is not a finding of fraud.",
    "fast_track": "Lower priority: expedite standard processing with coverage/document checks and safeguards. This is not fraud clearance or approval of payment.",
}


@lru_cache(maxsize=1)
def _load():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    path=ROOT/manifest['model_path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise RuntimeError("Locked model checksum mismatch")
    pipeline=joblib.load(path)
    prep=pipeline.named_steps['preprocessor']
    classifier=pipeline.named_steps['classifier']
    features=list(pipeline.feature_names_in_)
    numeric=list(prep.transformers_[0][2]); categorical=list(prep.transformers_[1][2])
    numeric_pipe=prep.named_transformers_['numeric']
    numeric_imputer=numeric_pipe.named_steps['imputer']
    cat_pipe=prep.named_transformers_['categorical']
    encoder=cat_pipe.named_steps['encoder']
    reference={f:float(v) for f,v in zip(numeric,numeric_imputer.statistics_)}
    for f,v in zip(categorical,cat_pipe.named_steps['imputer'].statistics_):
        reference[f]=v.item() if isinstance(v,np.generic) else v
    # Exact encoded ownership; include every one-hot level, even inactive ones.
    owners=list(numeric)
    for f,categories in zip(categorical,encoder.categories_):
        owners.extend([f]*len(categories))
    if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
        raise RuntimeError("Unsupported changed encoder grouping; revalidate ownership mapping")
    if len(owners)!=classifier.get_booster().num_features():
        raise RuntimeError("Encoded contribution ownership mismatch")
    return manifest,pipeline,features,reference,encoder,categorical,owners


def _missing(value):
    if value is None:
        return True
    if not isinstance(value,(str,Real,np.generic)):
        raise ValueError("Feature values must be scalar strings or numbers")
    return bool(pd.isna(value)) or (isinstance(value,str) and not value.strip())


def score(claim, exclude=None):
    """Score a flat claim; exclusions replace values with fitted training references.

    Native TreeSHAP `impact` values are in log-odds, not probability points.
    Missing/unknown fields are flagged; their model bands are provisional only.
    """
    if not isinstance(claim,dict):
        raise ValueError("claim must be a flat object")
    manifest,pipeline,features,reference,encoder,categorical,owners=_load()
    extra=set(claim)-set(features)-{'claim_id','exclude'}
    if extra:
        raise ValueError(f"Unexpected input fields (target must never be supplied): {sorted(extra)}")
    requested=claim.get('exclude',[]) if exclude is None else exclude
    if not isinstance(requested,(list,tuple)) or any(not isinstance(f,str) for f in requested):
        raise ValueError("exclude must be a list of original feature names")
    invalid=set(requested)-set(features)
    if invalid:
        raise ValueError(f"Unknown exclusion features: {sorted(invalid)}")
    excluded=[f for f in features if f in requested]
    normalized={};missing=[]
    for f in features:
        value=claim.get(f)
        if _missing(value):
            normalized[f]=np.nan;missing.append(f);continue
        if f=='Deductible' or f in INTEGER_CATEGORIES:
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
                raise ValueError(f"{f} must be finite numeric, not a numeric string")
            if f in INTEGER_CATEGORIES and float(value)!=int(value):
                raise ValueError(f"{f} must be an integer categorical level")
            normalized[f]=float(value) if f=='Deductible' else int(value)
        elif not isinstance(value,str):
            raise ValueError(f"{f} must be a source string category")
        else:
            normalized[f]=value
    unknown=[f for f,levels in zip(categorical,encoder.categories_)
             if f not in missing and normalized[f] not in levels]
    effective={**normalized,**{f:reference[f] for f in excluded}}
    frame=pd.DataFrame([effective],columns=features)
    prep=pipeline.named_steps['preprocessor']
    encoded=prep.transform(frame)
    booster=pipeline.named_steps['classifier'].get_booster()
    matrix=xgb.DMatrix(encoded,nthread=1)
    raw=booster.predict(matrix,pred_contribs=True,approx_contribs=False)[0]
    margin=float(booster.predict(matrix,output_margin=True)[0])
    probability=float(booster.predict(matrix)[0])
    aggregated={f:0.0 for f in features}
    for owner,value in zip(owners,raw[:-1]):
        aggregated[owner]+=float(value)
    bias=float(raw[-1])
    if not np.isclose(bias+sum(aggregated.values()),margin,atol=2e-5):
        raise RuntimeError("Native contribution additivity check failed")
    contributions=[]
    for f in features:
        value=effective[f]
        if f in missing and f not in excluded:
            value=reference[f]
        if isinstance(value,np.generic):
            value=value.item()
        contributions.append(dict(feature=f,label=LABELS[f],value=value,impact=aggregated[f],
                                  excluded=f in excluded,imputed=f in missing and f not in excluded))
    contributions.sort(key=lambda c:abs(c['impact']),reverse=True)
    tier='investigate' if probability>=manifest['investigate_min_score'] else 'review' if probability>=manifest['review_min_score'] else 'fast_track'
    sensitivity=None
    if excluded:
        original=float(pipeline.predict_proba(pd.DataFrame([normalized],columns=features))[0,1])
        sensitivity=dict(is_sensitivity_analysis=True,method="Replace excluded features with fitted train median/mode; no retraining or removal of encoded columns",
                         original_risk_score=original,score_change=probability-original,
                         reference_values={f:reference[f] for f in excluded},
                         warning="Hypothetical sensitivity result; must not replace the observed claim's operational priority")
    quality=bool(missing or unknown)
    recommendation=RECOMMENDATIONS[tier]
    if quality:
        recommendation="Input quality requires verification before operational routing. "+recommendation
    return dict(risk_score=probability,tier=tier,contributions=contributions,excluded=excluded,
                model_id=manifest['model_id'],score_type="relative fraud-risk ranking score; uncalibrated",
                priority_band={'investigate':'high','review':'medium','fast_track':'low'}[tier],
                tier_status="provisional_validation_distribution_band",recommendation=recommendation,
                contribution_method="XGBoost native exact TreeSHAP, summed over encoded columns per original feature",
                impact_unit="log_odds",base_value=bias,raw_margin=margin,
                sensitivity_analysis=sensitivity,missing_features=missing,unknown_categories=unknown,
                requires_data_quality_check=quality,
                operational_routing_allowed=not quality and not excluded,
                safety_note="Human decisions only; model flags do not confirm fraud or approve/reject claims. Bands do not enforce queue capacity.")
