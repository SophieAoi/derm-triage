"""Tests for the evaluation script. Run from the repo root: python3 -m pytest tests"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import condition_to_tier_probs, evaluate_run, tier_map_for, tier_metrics
from src.train import TrainConfig, train
from tests.test_dataset import SPLIT, TAXONOMY, SyntheticData


class TierMetricTests(unittest.TestCase):
    # Worked by hand. Confusion matrix (rows true, cols predicted Urgent, Soon, Routine):
    #   Urgent  [1, 0, 1]
    #   Soon    [0, 1, 0]
    #   Routine [0, 1, 2]
    Y_TRUE = [0, 0, 1, 2, 2, 2]
    Y_PRED = [0, 2, 1, 2, 1, 2]

    def setUp(self):
        self.m = tier_metrics(self.Y_TRUE, self.Y_PRED)

    def test_confusion_matrix(self):
        self.assertEqual(self.m["confusion_matrix"]["matrix"], [[1, 0, 1], [0, 1, 0], [0, 1, 2]])

    def test_sensitivity_and_specificity(self):
        expected = {"Urgent": (1 / 2, 4 / 4), "Soon": (1 / 1, 4 / 5), "Routine": (2 / 3, 2 / 3)}
        for tier, (sens, spec) in expected.items():
            with self.subTest(tier=tier):
                self.assertAlmostEqual(self.m["per_tier"][tier]["sensitivity"], sens)
                self.assertAlmostEqual(self.m["per_tier"][tier]["specificity"], spec)

    def test_summary_rates(self):
        self.assertAlmostEqual(self.m["accuracy"], 4 / 6)
        self.assertAlmostEqual(self.m["balanced_accuracy"], (1 / 2 + 1 + 2 / 3) / 3)
        self.assertAlmostEqual(self.m["undertriage_rate"], 1 / 6)  # Urgent predicted Routine
        self.assertAlmostEqual(self.m["overtriage_rate"], 1 / 6)   # Routine predicted Soon

    def test_tier_with_no_images_has_no_sensitivity(self):
        m = tier_metrics([0, 2], [0, 2])
        self.assertIsNone(m["per_tier"]["Soon"]["sensitivity"])
        self.assertEqual(m["balanced_accuracy"], 1.0)


class MappingTests(unittest.TestCase):
    def test_condition_probabilities_summed_into_tiers(self):
        tier_map = {"bcc": "Soon", "mel": "Urgent", "nv": "Routine", "df": "Routine"}
        probs = np.array([[0.1, 0.2, 0.3, 0.4]])
        tiers = condition_to_tier_probs(probs, ["bcc", "mel", "nv", "df"], tier_map)
        np.testing.assert_allclose(tiers, [[0.2, 0.1, 0.7]])

    def test_alternative_mappings_override_taxonomy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "taxonomy.csv"
            pd.DataFrame([("ham10000", "akiec", "Soon"), ("ham10000", "bcc", "Soon"),
                          ("ham10000", "mel", "Urgent")],
                         columns=["dataset", "source_label", "urgency_tier"]).to_csv(path, index=False)
            self.assertEqual(tier_map_for("ham10000", "default", path)["akiec"], "Soon")
            cautious = tier_map_for("ham10000", "cautious", path)
            self.assertEqual((cautious["akiec"], cautious["bcc"]), ("Urgent", "Urgent"))
            self.assertEqual(tier_map_for("ham10000", "relaxed", path)["akiec"], "Routine")


class EvaluateRunTests(unittest.TestCase):
    def setUp(self):
        self.data = SyntheticData(SPLIT, TAXONOMY)
        split_dir = self.data.splits_dir / "ham10000"
        for split in ("val", "test"):
            shutil.copy(split_dir / "train.csv", split_dir / f"{split}.csv")
        self.runs = tempfile.TemporaryDirectory()
        self.paths = {"data_dir": self.data.data_dir, "splits_dir": self.data.splits_dir,
                      "taxonomy_path": self.data.taxonomy}

    def tearDown(self):
        self.data.close()
        self.runs.cleanup()

    def trained(self, target):
        config = TrainConfig(target=target, arch="resnet18", pretrained=False, epochs=1,
                             batch_size=2, num_workers=0, device="cpu")
        return train(config, runs_dir=Path(self.runs.name), **self.paths)

    def evaluate(self, run_dir, **kwargs):
        return evaluate_run(run_dir, device="cpu", num_workers=0, **self.paths, **kwargs)

    def test_tier_model(self):
        metrics, out_dir = self.evaluate(self.trained("tier"), split="test")
        self.assertEqual(out_dir.name, "eval_test_default")
        self.assertEqual(metrics["tier"]["n"], 4)
        predictions = pd.read_csv(out_dir / "predictions.csv")
        self.assertEqual(list(predictions["true_tier"]), ["Urgent", "Routine", "Routine", "Soon"])
        np.testing.assert_allclose(predictions[["p_urgent", "p_soon", "p_routine"]].sum(axis=1), 1, atol=1e-5)
        self.assertEqual(json.loads((out_dir / "metrics.json").read_text())["split"], "test")

    def test_condition_model_reports_both_tier_methods(self):
        metrics, out_dir = self.evaluate(self.trained("condition"))
        self.assertIn("tier", metrics)
        self.assertIn("tier_from_top_condition", metrics)
        self.assertEqual(metrics["condition"]["confusion_matrix"]["rows_true_cols_predicted"], ["bcc", "mel", "nv"])
        predictions = pd.read_csv(out_dir / "predictions.csv")
        self.assertIn("predicted_condition", predictions)

    def test_condition_model_under_cautious_mapping(self):
        metrics, out_dir = self.evaluate(self.trained("condition"), mapping="cautious")
        self.assertEqual(out_dir.name, "eval_val_cautious")
        predictions = pd.read_csv(out_dir / "predictions.csv")
        self.assertEqual(list(predictions["true_tier"]), ["Urgent", "Routine", "Routine", "Urgent"])  # bcc -> Urgent

    def test_tier_model_rejects_other_mappings(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.trained("tier"), mapping="cautious")

    def test_only_val_and_test_splits_allowed(self):
        run_dir = self.trained("tier")
        for split in ("train", "ddi"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.evaluate(run_dir, split=split)


if __name__ == "__main__":
    unittest.main()
