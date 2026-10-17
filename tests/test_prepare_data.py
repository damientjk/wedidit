"""Run: python -m unittest discover tests  (stdlib only, s390x-safe)."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_data import LABEL, RAW_PATH, load, prepare, write_splits  # noqa: E402
from synth_amount import COLLISIONS, synthesise  # noqa: E402

SYNTH = ["incident_type(synth)", "accident_severity(synth)", "claim_amount_usd(synth)"]


@unittest.skipUnless(RAW_PATH.exists(), "raw CSV not present")
class PrepareDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load(RAW_PATH)
        cls.df = prepare(cls.raw)

    def test_drops_only_the_invalid_date_row(self):
        self.assertEqual(len(self.df), len(self.raw) - 1)

    def test_no_zero_age_and_flag_counts_match(self):
        self.assertGreaterEqual(self.df["Age"].min(), 16)
        self.assertEqual(self.df["age_was_missing"].sum(), (self.raw["Age"] == 0).sum() - 1)

    def test_drops_policytype_and_renames_id(self):
        self.assertNotIn("PolicyType", self.df.columns)
        self.assertTrue(self.df["claim_id"].is_unique)

    def test_split_shares_and_fraud_rate_are_stratified(self):
        shares = self.df["split"].value_counts(normalize=True)
        self.assertAlmostEqual(shares["train"], 0.6, delta=0.005)
        rates = self.df.groupby("split")[LABEL].mean()
        self.assertLess(rates.max() - rates.min(), 0.002)

    def test_split_is_reproducible(self):
        self.assertTrue(prepare(self.raw)["split"].equals(self.df["split"]))

    def test_split_files_partition_the_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            parts = [pd.read_csv(p) for p in write_splits(self.df, Path(tmp))]
        self.assertEqual(sum(len(p) for p in parts), len(self.df))
        ids = pd.concat(p["claim_id"] for p in parts)
        self.assertTrue(ids.is_unique)
        for part, name in zip(parts, ["train", "val", "test"]):
            self.assertEqual(set(part["split"]), {name})

    def test_synth_rejects_label(self):
        with self.assertRaises(ValueError):
            synthesise(self.df)

    def test_synth_is_reproducible_from_features_alone(self):
        features = self.df.drop(columns=[LABEL, *SYNTH, "split"])
        self.assertTrue(synthesise(features).equals(self.df[SYNTH]))

    def test_amount_is_plausible(self):
        amt = self.df["claim_amount_usd(synth)"]
        self.assertTrue((amt >= 0).all())
        self.assertLess(amt.max(), 150_000)
        self.assertTrue(1_000 < amt.median() < 4_000)
        by_cov = self.df.groupby("BasePolicy")["claim_amount_usd(synth)"].median()
        self.assertGreater(by_cov["All Perils"], by_cov["Liability"])

    def test_third_party_fault_liability_claims_are_small(self):
        mask = (self.df["BasePolicy"] == "Liability") & (self.df["Fault"] == "Third Party")
        self.assertLess(self.df.loc[mask, "claim_amount_usd(synth)"].median(), 800)

    def test_comprehensive_incidents_only_under_all_perils(self):
        comprehensive = ~self.df["incident_type(synth)"].isin(COLLISIONS)
        self.assertTrue((self.df.loc[comprehensive, "BasePolicy"] == "All Perils").all())
        self.assertTrue((self.df.loc[comprehensive, "Fault"] == "Policy Holder").all())

    def test_other_party_at_fault_means_two_vehicle_incident(self):
        tp = self.df["Fault"] == "Third Party"
        allowed = {"Collision - other vehicle", "Hit while parked"}
        self.assertTrue(set(self.df.loc[tp, "incident_type(synth)"]) <= allowed)

    def test_severity_is_consistent(self):
        sev = self.df["accident_severity(synth)"]
        self.assertEqual(set(sev), {"Minor", "Moderate", "Major", "Total loss"})
        glass = self.df["incident_type(synth)"] == "Glass only"
        self.assertTrue(sev[glass].isin(["Minor", "Moderate"]).all())
        own = self.df["BasePolicy"] != "Liability"
        med = self.df[own].groupby(sev[own])["claim_amount_usd(synth)"].median()
        self.assertTrue(med["Minor"] < med["Moderate"] < med["Major"])

if __name__ == "__main__":
    unittest.main()
