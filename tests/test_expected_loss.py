"""Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expected_loss import (  # noqa: E402
    AMOUNT, LABEL, apply_platt, cross_fitted_probability, fit_platt, ranking_comparison,
)


def _inflated_scores(n=4000, seed=0):
    """True probability p, but the 'model' reports an inflated sqrt(p), like a class-weighted XGBoost."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 15, n)
    labels = pd.Series((rng.random(n) < p).astype(int))
    return pd.Series(np.sqrt(p)), labels, p


class CalibrationTest(unittest.TestCase):
    def test_platt_pulls_inflated_scores_back_to_true_rate(self):
        scores, labels, _ = _inflated_scores()
        calibrated = apply_platt(fit_platt(scores, labels), scores)
        self.assertGreater(scores.mean(), 3 * labels.mean())
        self.assertAlmostEqual(calibrated.mean(), labels.mean(), delta=0.005)

    def test_calibration_keeps_ranking(self):
        scores, labels, _ = _inflated_scores()
        calibrated = apply_platt(fit_platt(scores, labels), scores)
        self.assertTrue((np.argsort(calibrated) == np.argsort(scores.to_numpy())).all())

    def test_cross_fitted_probability_covers_every_row(self):
        scores, labels, _ = _inflated_scores()
        prob = cross_fitted_probability(scores, labels)
        self.assertFalse(prob.isna().any())


class RankingTest(unittest.TestCase):
    def test_expected_loss_prefers_large_likely_claims(self):
        df = pd.DataFrame({LABEL: [1, 1, 0, 0], AMOUNT: [300, 15_000, 100, 100]})
        prob = pd.Series([0.6, 0.3, 0.1, 0.1])
        table = ranking_comparison(df, prob, capacity=0.25).set_index("ranked_by")
        self.assertEqual(table.loc["probability", "fraud_usd_caught"], 300)
        self.assertEqual(table.loc["expected_loss", "fraud_usd_caught"], 15_000)
        self.assertEqual(table.loc["expected_loss", "fraud_cases_caught"], 1)


if __name__ == "__main__":
    unittest.main()
