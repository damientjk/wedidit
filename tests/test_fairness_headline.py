"""Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairness import group_rates  # noqa: E402
from headline import AMOUNT, LABEL, bootstrap, headline_metrics  # noqa: E402


class FairnessTest(unittest.TestCase):
    def setUp(self):
        n = 40
        self.groups = pd.Series(["a"] * n + ["b"] * n)
        # Group b is flagged twice as often; both groups have the same fraud.
        self.tiers = pd.Series(["review"] * 10 + ["fast_track"] * 30 + ["review"] * 20 + ["fast_track"] * 20)
        self.labels = pd.Series(([1] * 10 + [0] * 30) * 2)

    def test_flag_rate_and_recall_per_group(self):
        t = group_rates(self.tiers, self.labels, self.groups)
        self.assertAlmostEqual(t.loc["a", "flag_rate"], 0.25)
        self.assertAlmostEqual(t.loc["b", "flag_rate"], 0.5)
        self.assertAlmostEqual(t.loc["a", "recall"], 1.0)

    def test_small_groups_are_marked(self):
        t = group_rates(self.tiers[:50], self.labels[:50], self.groups[:50])
        self.assertEqual(t.loc["b", "note"], "too small to judge")


class HeadlineTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "tier": ["fast_track"] * 6 + ["review", "investigate", "review", "investigate"],
            LABEL: [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            AMOUNT: [100] * 5 + [500, 1_000, 2_000, 100, 4_000],
        })

    def test_point_estimates(self):
        m = headline_metrics(self.df)
        self.assertAlmostEqual(m["share_fast_tracked"], 0.6)
        self.assertAlmostEqual(m["share_fraud_to_human"], 0.75)
        self.assertAlmostEqual(m["share_fraud_usd_to_human"], 7_000 / 7_500)

    def test_interval_contains_estimate_and_is_reproducible(self):
        big = pd.concat([self.df] * 30, ignore_index=True)
        table = bootstrap(big, n_boot=200)
        row = table.loc["share_fraud_to_human"]
        self.assertLessEqual(row.ci_low, row.estimate)
        self.assertGreaterEqual(row.ci_high, row.estimate)
        self.assertTrue(table.equals(bootstrap(big, n_boot=200)))


if __name__ == "__main__":
    unittest.main()
