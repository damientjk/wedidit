"""Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk import (  # noqa: E402
    AMOUNT, LABEL, cost_optimal_cutoff, fast_track_cutoff, investigate_cutoff, make_tier, triage_curve,
)

# The 10-claim example from the walkthrough: fraud at ranks 5, 8 and 10.
SCORES = pd.Series([0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.15, 0.30, 0.45, 0.70])
FRAUD = pd.Series([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])


class TriageCurveTest(unittest.TestCase):
    def test_counts_fraud_below_each_cutoff(self):
        curve = triage_curve(SCORES, FRAUD)
        self.assertEqual(list(curve["share_fraud_leaked"].round(2)),
                         [0, 0, 0, 0, 0.33, 0.33, 0.33, 0.67, 0.67, 1.0])
        self.assertAlmostEqual(curve["share_fast_tracked"].iloc[3], 0.4)

    def test_order_of_input_does_not_matter(self):
        shuffled = SCORES.sample(frac=1, random_state=0)
        curve = triage_curve(shuffled, FRAUD[shuffled.index])
        self.assertTrue(curve.equals(triage_curve(SCORES, FRAUD)))

    def test_tied_scores_are_not_split(self):
        curve = triage_curve(pd.Series([0.1, 0.1, 0.2]), pd.Series([0, 1, 0]))
        self.assertEqual(len(curve), 2)


class CutoffTest(unittest.TestCase):
    def test_fast_track_cutoff_respects_leakage_limit(self):
        curve = triage_curve(SCORES, FRAUD)
        a = fast_track_cutoff(curve, max_leaked=0.10)
        self.assertEqual(a, 0.07)  # fast-track the four claims below 0.07, no fraud leaked
        self.assertEqual(FRAUD[SCORES < a].sum(), 0)

    def test_investigate_cutoff_takes_top_share(self):
        b = investigate_cutoff(SCORES, capacity=0.2)
        self.assertEqual((SCORES >= b).sum(), 2)

    def test_investigate_cutoff_rejects_bad_capacity(self):
        with self.assertRaises(ValueError):
            investigate_cutoff(SCORES, capacity=1.5)

    def test_cheap_investigations_push_cutoff_down(self):
        df = pd.DataFrame({"score": SCORES, LABEL: FRAUD, AMOUNT: 1_000})
        self.assertLess(cost_optimal_cutoff(df, 10), cost_optimal_cutoff(df, 2_000))

    def test_tier_rule_matches_api_semantics(self):
        tier = make_tier(0.05, 0.3)
        self.assertEqual([tier(0.01), tier(0.05), tier(0.29), tier(0.3)],
                         ["fast_track", "review", "review", "investigate"])


if __name__ == "__main__":
    unittest.main()
