"""Tests for the uncertainty code. Run from the repo root: python3 -m pytest tests"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import TrainConfig, train
from src.uncertainty import (
    analyse_run,
    entropy,
    error_on_most_uncertain,
    escalation_metrics,
    escalation_sweep,
    expected_calibration_error,
    fit_temperature,
    one_minus_max_softmax,
    risk_coverage,
    softmax,
)
from tests.test_dataset import SPLIT, TAXONOMY, SyntheticData

U, S, R = 0, 1, 2  # Urgent, Soon, Routine


def random_cases(n=500, seed=0):
    rng = np.random.default_rng(seed)
    true_tier = rng.choice([U, S, R], size=n, p=[0.2, 0.2, 0.6])
    pred_tier = np.where(rng.random(n) < 0.7, true_tier, rng.integers(0, 3, n))
    uncertainty = rng.random(n) * np.log(3)
    return true_tier, pred_tier, uncertainty


class EntropyTests(unittest.TestCase):
    def test_zero_for_one_hot(self):
        np.testing.assert_allclose(entropy(np.eye(7)), 0.0)

    def test_maximal_for_uniform(self):
        for k in (3, 7):
            with self.subTest(k=k):
                uniform = np.full((1, k), 1 / k)
                self.assertAlmostEqual(entropy(uniform)[0], np.log(k))
                rng = np.random.default_rng(k)
                others = rng.dirichlet(np.ones(k), size=1000)
                self.assertTrue((entropy(others) <= np.log(k) + 1e-12).all())

    def test_one_minus_max_softmax(self):
        np.testing.assert_allclose(one_minus_max_softmax([[0.7, 0.2, 0.1], [1 / 3] * 3]), [0.3, 2 / 3])


class EscalationTests(unittest.TestCase):
    def test_escalation_rate_rises_as_threshold_falls(self):
        sweep = escalation_sweep(*random_cases(), n_classes=3, score="entropy")
        rates = sweep.sort_values("threshold", ascending=False)["escalation_rate"].to_numpy()
        self.assertTrue((np.diff(rates) >= 0).all())
        self.assertEqual(rates[0], 0.0)  # at the maximum threshold nothing is escalated
        self.assertGreater(rates[-1], 0.99)  # at zero almost everything is

    def test_urgent_sensitivity_never_falls_with_more_escalation(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                sweep = escalation_sweep(*random_cases(seed=seed), n_classes=3, score="entropy")
                ordered = sweep.sort_values("escalation_rate", kind="stable")["urgent_sensitivity"].to_numpy()
                self.assertTrue((np.diff(ordered) >= -1e-12).all())

    def test_counts_on_hand_worked_example(self):
        true_tier = np.array([U, U, U, S, R, R])
        pred_tier = np.array([U, R, S, S, R, U])
        uncertainty = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
        m = escalation_metrics(true_tier, pred_tier, uncertainty, threshold=0.5)
        # Escalated: cases 1 (missed Urgent, caught), 3 (correct Soon), 4 (correct Routine).
        self.assertEqual(m["escalated"], 3)
        self.assertAlmostEqual(m["escalation_rate"], 0.5)
        self.assertAlmostEqual(m["urgent_sensitivity"], 2 / 3)  # case 2 (Urgent -> Soon) still missed
        self.assertEqual(m["missed_urgent_caught"], 1)
        self.assertEqual(m["missed_urgent_still_missed"], 1)
        self.assertEqual(m["correct_sent_up_needlessly"], 2)
        self.assertEqual(m["undertriage_urgent_to_routine"], 0)
        self.assertEqual(m["overtriage_to_urgent"], 3)  # cases 3, 4, 5 (5 was predicted Urgent already)
        self.assertAlmostEqual(m["overtriage_rate"], 1.0)


class RiskCoverageTests(unittest.TestCase):
    def test_hand_worked_curve(self):
        errors = np.array([0, 1, 0, 1])
        uncertainty = np.array([0.1, 0.4, 0.2, 0.3])  # keep order: 0, 2, 3, 1
        coverage, risk, aurc = risk_coverage(errors, uncertainty)
        np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(risk, [0, 0, 1 / 3, 1 / 2])
        self.assertAlmostEqual(aurc, (0 + 0 + 1 / 3 + 1 / 2) / 4)

    def test_error_on_most_uncertain(self):
        errors = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        uncertainty = np.arange(10)
        self.assertEqual(error_on_most_uncertain(errors, uncertainty, 0.1), 0.0)  # case 9 only
        self.assertEqual(error_on_most_uncertain(errors, uncertainty, 0.2), 0.5)  # cases 8 and 9


class CalibrationTests(unittest.TestCase):
    def test_ece_zero_when_perfectly_calibrated(self):
        # 10 cases at confidence 0.8 with 8 correct, 10 at 0.6 with 6 correct.
        probs = np.array([[0.8, 0.2]] * 10 + [[0.6, 0.4]] * 10)
        labels = np.array([0] * 8 + [1] * 2 + [0] * 6 + [1] * 4)
        self.assertAlmostEqual(expected_calibration_error(probs, labels), 0.0)

    def test_ece_on_overconfident_example(self):
        probs = np.array([[0.9, 0.1]] * 10)
        labels = np.array([0] * 5 + [1] * 5)  # 90% confident, 50% right
        self.assertAlmostEqual(expected_calibration_error(probs, labels), 0.4)

    def test_temperature_recovers_known_value(self):
        rng = np.random.default_rng(0)
        true_logits = rng.normal(size=(20000, 3)) * 2
        labels = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
        fitted = fit_temperature(true_logits * 2.5, labels)  # model 2.5x overconfident
        self.assertAlmostEqual(fitted, 2.5, delta=0.1)


class AnalyseRunTests(unittest.TestCase):
    def test_writes_outputs_without_touching_run_files(self):
        data = SyntheticData(SPLIT, TAXONOMY)
        runs = tempfile.TemporaryDirectory()
        try:
            split_dir = data.splits_dir / "ham10000"
            shutil.copy(split_dir / "train.csv", split_dir / "val.csv")
            paths = {"data_dir": data.data_dir, "splits_dir": data.splits_dir, "taxonomy_path": data.taxonomy}
            for target in ("tier", "condition"):
                with self.subTest(target=target):
                    config = TrainConfig(target=target, arch="resnet18", pretrained=False, epochs=1,
                                         batch_size=2, num_workers=0, device="cpu")
                    run_dir = train(config, runs_dir=Path(runs.name), **paths)
                    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
                    analyse_run(run_dir, device="cpu", num_workers=0, **paths)
                    after = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
                    self.assertEqual(before, after)
                    out = run_dir / "uncertainty_val"
                    self.assertEqual({p.name for p in out.iterdir()},
                                     {"predictions.csv", "escalation_sweep.csv", "risk_coverage.csv",
                                      "reliability_tier_level.csv", "summary.json"})
                    table = pd.read_csv(out / "predictions.csv")
                    self.assertEqual(len(table), 4)
                    np.testing.assert_allclose(table[["p_urgent", "p_soon", "p_routine"]].sum(axis=1), 1, atol=1e-5)
                    self.assertEqual(len(pd.read_csv(out / "escalation_sweep.csv")), 200)  # 100 per score
        finally:
            data.close()
            runs.cleanup()


if __name__ == "__main__":
    unittest.main()
