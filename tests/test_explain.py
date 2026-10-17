"""Inference contract tests; no fitting or test.csv access."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import explain


class ExplainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.pipeline,cls.features,cls.reference,*_=explain._load()
        cls.rows=pd.read_csv(explain.ROOT/'data/processed/val.csv').head(12)
        cls.claim=cls.rows[cls.features].iloc[0].to_dict()

    def test_real_claim_pipeline_parity_and_original_feature_additivity(self):
        for _,row in self.rows.iterrows():
            claim=row[self.features].to_dict()
            result=explain.score(claim)
            expected=self.pipeline.predict_proba(pd.DataFrame([claim],columns=self.features))[0,1]
            self.assertAlmostEqual(result['risk_score'],expected,places=6)
            self.assertEqual({c['feature'] for c in result['contributions']},set(self.features))
            margin=result['base_value']+sum(c['impact'] for c in result['contributions'])
            self.assertAlmostEqual(margin,result['raw_margin'],places=4)
            self.assertAlmostEqual(1/(1+np.exp(-margin)),result['risk_score'],places=6)
            json.dumps(result,allow_nan=False)

    def test_missing_uses_fitted_imputers_and_blocks_operational_routing(self):
        claim={**self.claim,'Make':None,'Deductible':None}
        result=explain.score(claim)
        self.assertCountEqual(result['missing_features'],['Make','Deductible'])
        self.assertFalse(result['operational_routing_allowed'])
        imputed={**self.claim,'Make':self.reference['Make'],'Deductible':self.reference['Deductible']}
        self.assertAlmostEqual(result['risk_score'],explain.score(imputed)['risk_score'],places=6)
        empty=explain.score({})
        self.assertEqual(len(empty['missing_features']),19)
        self.assertTrue(empty['requires_data_quality_check'])

    def test_unknown_categories(self):
        result=explain.score({**self.claim,'Make':'UNSEEN_BRAND'})
        self.assertIn('Make',result['unknown_categories'])
        self.assertTrue(np.isfinite(result['risk_score']))
        self.assertFalse(result['operational_routing_allowed'])

    def test_exclusions_are_reference_replacements_without_mutation(self):
        claim=copy.deepcopy(self.claim); before=copy.deepcopy(claim)
        result=explain.score(claim,exclude=['Make','Deductible','Make'])
        self.assertEqual(claim,before)
        self.assertEqual(len(result['excluded']),2)
        modified={**claim,'Make':self.reference['Make'],'Deductible':self.reference['Deductible']}
        self.assertAlmostEqual(result['risk_score'],explain.score(modified)['risk_score'],places=6)
        self.assertTrue(result['sensitivity_analysis']['is_sensitivity_analysis'])
        self.assertFalse(result['operational_routing_allowed'])
        self.assertEqual(sum(c['excluded'] for c in result['contributions']),2)

    def test_all_excluded_and_embedded_payload(self):
        result=explain.score({**self.claim,'exclude':self.features})
        self.assertEqual(len(result['excluded']),19)
        self.assertAlmostEqual(result['risk_score'],explain.score(self.reference)['risk_score'],places=6)
        json.dumps(result,allow_nan=False)

    def test_invalid_fields_types_and_exclusions(self):
        for claim in ({**self.claim,'FraudFound_P':0}, {**self.claim,'Deductible':float('inf')},
                      {**self.claim,'DriverRating':'1'}, {**self.claim,'Make':[]},
                      {**self.claim,'WeekOfMonth':1.5}):
            with self.assertRaises(ValueError):explain.score(claim)
        for excluded in ('Make',['FraudFound_P'],[None]):
            with self.assertRaises(ValueError):explain.score(self.claim,exclude=excluded)

    def test_no_runtime_csv_reads_or_fitting(self):
        with patch('pandas.read_csv',side_effect=AssertionError('Runtime dataset access forbidden')):
            with patch.object(type(self.pipeline),'fit',side_effect=AssertionError('Fitting forbidden')):
                result=explain.score(self.claim)
                self.assertTrue(result['operational_routing_allowed'])

    def test_bands_match_defined_score_boundaries(self):
        result=explain.score(self.claim)
        p=result['risk_score']
        expected='investigate' if p>=self.manifest['investigate_min_score'] else 'review' if p>=self.manifest['review_min_score'] else 'fast_track'
        self.assertEqual(result['tier'],expected)
        self.assertEqual(result['tier_status'],'provisional_validation_distribution_band')
        self.assertEqual(result['impact_unit'],'log_odds')


if __name__=='__main__':
    unittest.main()
