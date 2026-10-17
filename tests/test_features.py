"""
tests/test_features.py (owner: Ethan)
Run with:  python -m unittest discover tests
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import FEATURE_NAMES, FORBIDDEN, build_features, check_columns  # noqa: E402

CLAIM = {
    "Month": "Dec", "WeekOfMonth": 5, "DayOfWeek": "Wednesday", "Make": "Honda",
    "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
    "WeekOfMonthClaimed": 1, "Sex": "Female", "MaritalStatus": "Single", "Age": 21,
    "Fault": "Policy Holder", "VehicleCategory": "Sport", "VehiclePrice": "more than 69000",
    "Deductible": 300, "DriverRating": 1, "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
    "AgeOfVehicle": "3 years", "AgeOfPolicyHolder": "26 to 30", "PoliceReportFiled": "No",
    "WitnessPresent": "No", "AgentType": "External", "NumberOfSuppliments": "none",
    "AddressChange_Claim": "1 year", "NumberOfCars": "3 to 4", "BasePolicy": "All Perils",
    "age_was_missing": 0,
}


class FeatureTests(unittest.TestCase):
    def test_single_claim(self):
        X = build_features(pd.DataFrame([CLAIM]))
        self.assertEqual(X.shape, (1, len(FEATURE_NAMES)))
        self.assertEqual(list(X.columns), FEATURE_NAMES)

    def test_wraparound_month(self):
        X = build_features(pd.DataFrame([CLAIM]))
        self.assertEqual(X.loc[0, "months_to_claim"], 1)  # Dec -> Jan

    def test_unknown_value_becomes_nan_not_crash(self):
        bad = {**CLAIM, "PastNumberOfClaims": "banana"}
        X = build_features(pd.DataFrame([bad]))
        self.assertTrue(pd.isna(X.loc[0, "past_claims"]))

    def test_missing_column_becomes_nan(self):
        partial = {k: v for k, v in CLAIM.items() if k != "Fault"}
        X = build_features(pd.DataFrame([partial]))
        self.assertTrue(pd.isna(X.loc[0, "fault_policy_holder"]))
        self.assertIn("Fault", check_columns(pd.DataFrame([partial])))

    def test_old_brand_spelling_accepted(self):
        X = build_features(pd.DataFrame([{**CLAIM, "Make": "Porche"}]))
        self.assertEqual(X.loc[0, "Make_Porsche"], 1.0)

    def test_no_forbidden_or_synthetic_features(self):
        for f in FEATURE_NAMES:
            self.assertNotIn(f, FORBIDDEN)
            self.assertFalse(f.endswith("(synth)"))

    def test_same_result_one_row_vs_batch(self):
        batch = build_features(pd.DataFrame([CLAIM, CLAIM]))
        single = build_features(pd.DataFrame([CLAIM]))
        pd.testing.assert_series_equal(batch.iloc[0], single.iloc[0], check_names=False)


if __name__ == "__main__":
    unittest.main()
