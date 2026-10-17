"""Experimental inference adapter; no API/deployment changes.

Score immediately, finalize tiers only for a closed, explicitly budgeted batch.
Explanations are feature occlusion score changes, not causal or SHAP values.
"""
from pathlib import Path
from numbers import Integral

import joblib
import numpy as np
import pandas as pd

from train import FEATURES, NUMERIC_FEATURES

TEXT = {
    "investigate": "Prioritize for specialist investigation. The risk flag is not a finding of fraud; a human must assess the evidence.",
    "review": "Request lightweight human verification before progressing through standard processing. The risk flag is not a finding of fraud.",
    "fast_track": "Expedite standard processing with coverage/document checks and applicable safeguards. This does not certify absence of fraud or authorize payment.",
}


class TriageCandidate:
    def __init__(self, model_path, reference_claim):
        self.path=Path(model_path)
        self.model=joblib.load(self.path)
        self.reference=reference_claim
        if list(self.model.feature_names_in_)!=FEATURES:
            raise ValueError("Unexpected feature schema")

    def score(self, claims, explain=True):
        """claims = [{claim_id, features: {all 19 fields}}]. No target accepted."""
        ids=[str(c['claim_id']) for c in claims]
        if not claims:
            return []
        try:
            [int(value) for value in ids]
        except ValueError as error:
            raise ValueError("claim_id must support the numeric tie-break") from error
        if len(set(ids))!=len(ids):
            raise ValueError("Duplicate claim IDs")
        rows=[]
        for claim in claims:
            fields=claim['features']
            if set(fields)!=set(FEATURES):
                raise ValueError("Exactly the 19 approved feature keys are required; no target/extra fields")
            if any(pd.isna(fields[f]) for f in FEATURES):
                raise ValueError("Incomplete inputs: use the standard data-quality exception workflow")
            if not isinstance(fields['Deductible'],(int,float)) or isinstance(fields['Deductible'],bool) or not np.isfinite(fields['Deductible']):
                raise ValueError("Deductible must be finite numeric")
            for f in ('WeekOfMonth','WeekOfMonthClaimed','DriverRating'):
                if not isinstance(fields[f],Integral) or isinstance(fields[f],bool):
                    raise ValueError(f"{f} must retain its integer categorical input type")
            for f in set(FEATURES)-{'Deductible','WeekOfMonth','WeekOfMonthClaimed','DriverRating'}:
                if not isinstance(fields[f],str):
                    raise ValueError(f"{f} must be a source string category")
            rows.append({f:fields[f] for f in FEATURES})
        x=pd.DataFrame(rows,columns=FEATURES)
        scores=self.model.predict_proba(x)[:,1]
        prep=self.model.named_steps['preprocessor']
        category_pipe=prep.named_transformers_['categorical']
        encoder=category_pipe.named_steps.get('encoder') or category_pipe.named_steps.get('onehot')
        categorical=[f for f in FEATURES if f not in NUMERIC_FEATURES]
        output=[]
        for i,score in enumerate(scores):
            unknown=[f for f,allowed in zip(categorical,encoder.categories_) if x.iloc[i][f] not in allowed]
            contributions=[]
            if explain:
                altered=pd.concat([x.iloc[[i]]]*len(FEATURES),ignore_index=True)
                for j,f in enumerate(FEATURES):
                    altered.loc[j,f]=self.reference[f]
                changed=self.model.predict_proba(altered)[:,1]
                contributions=sorted([
                    {"feature":f,"value":rows[i][f],"reference_value":self.reference[f],
                     "score_delta_on_restoring_observed_value":float(score-changed[j])}
                    for j,f in enumerate(FEATURES)],key=lambda v:abs(v['score_delta_on_restoring_observed_value']),reverse=True)[:5]
            output.append({"claim_id":ids[i],"model_id":self.path.stem,
                "risk_score":float(score),"score_type":"uncalibrated fraud-risk ranking score",
                "tier":None,"tier_status":"pending_closed_batch_ranking",
                "recommendation":"Await capacity-based queue assignment; continue standard processing safeguards.",
                "top_contributing_features":contributions,"explanation_method":"one-feature-at-a-time replacement with training median/mode; non-additive, non-causal",
                "unknown_categories":unknown,"requires_data_quality_check":bool(unknown)})
        return output

    @staticmethod
    def assign_batch(scored, investigate_slots, review_slots):
        if not (isinstance(investigate_slots,int) and isinstance(review_slots,int)) or min(investigate_slots,review_slots)<0 or investigate_slots+review_slots>len(scored):
            raise ValueError("Invalid explicit batch slot counts")
        if len({r['claim_id'] for r in scored})!=len(scored):
            raise ValueError("Duplicate claim IDs")
        if len({r['model_id'] for r in scored})>1:
            raise ValueError("One model version is required per batch")
        # Match the numeric claim-ID tie-break used by offline evaluation.
        ranked=sorted(scored,key=lambda r:(-r['risk_score'],int(r['claim_id'])))
        result=[]
        for i,row in enumerate(ranked):
            tier='investigate' if i<investigate_slots else 'review' if i<investigate_slots+review_slots else 'fast_track'
            result.append({**row,"rank":i+1,"tier":tier,"tier_status":"closed_batch_assignment",
                           "recommendation":TEXT[tier],
                           "safeguard_note":"Mandatory referrals, data-quality exceptions and audits require separate capacity accounting; this adapter does not silently override queues."})
        return result
