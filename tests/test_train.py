"""Smoke tests for the training loop. Run from the repo root: python3 -m pytest tests

Uses a tiny synthetic dataset, an untrained ResNet-18 and the CPU, so it is quick
and never downloads weights.
"""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.model import build_model
from src.train import TrainConfig, train
from tests.test_dataset import SPLIT, TAXONOMY, SyntheticData

CONFIG = TrainConfig(arch="resnet18", pretrained=False, epochs=2, batch_size=2,
                     num_workers=0, device="cpu", seed=7)


class TrainLoopTests(unittest.TestCase):
    def setUp(self):
        self.data = SyntheticData(SPLIT, TAXONOMY)
        split_dir = self.data.splits_dir / "ham10000"
        shutil.copy(split_dir / "train.csv", split_dir / "val.csv")
        self.runs = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.data.close()
        self.runs.cleanup()

    def run_training(self, config=CONFIG):
        return train(config, data_dir=self.data.data_dir, splits_dir=self.data.splits_dir,
                     taxonomy_path=self.data.taxonomy, runs_dir=Path(self.runs.name))

    def read_metrics(self, run_dir):
        with open(run_dir / "metrics.csv") as f:
            return list(csv.DictReader(f))

    def test_run_folder_contents(self):
        run_dir = self.run_training()
        self.assertEqual({p.name for p in run_dir.iterdir()},
                         {"config.json", "metrics.csv", "best.pt", "summary.json"})

        config = json.loads((run_dir / "config.json").read_text())
        self.assertEqual(config["config"]["seed"], 7)
        self.assertEqual(config["classes"], ["Urgent", "Soon", "Routine"])
        self.assertEqual(config["train_class_counts"], [1, 1, 2])
        self.assertIn("commit", config["git"])

        self.assertEqual([r["epoch"] for r in self.read_metrics(run_dir)], ["1", "2"])
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertIn(summary["best_epoch"], (1, 2))

    def test_checkpoint_reloads(self):
        checkpoint = torch.load(self.run_training() / "best.pt", weights_only=True)
        model = build_model(len(checkpoint["classes"]), arch="resnet18", pretrained=False)
        model.load_state_dict(checkpoint["model_state"])

    def test_same_seed_gives_same_metrics(self):
        def metrics_without_time(run_dir):
            return [{k: v for k, v in row.items() if k != "seconds"} for row in self.read_metrics(run_dir)]
        first = metrics_without_time(self.run_training())
        second = metrics_without_time(self.run_training())
        self.assertEqual(first, second)

    def test_runs_started_in_the_same_second_get_separate_folders(self):
        from src.train import make_run_dir
        runs = Path(self.runs.name)
        first, second = make_run_dir(runs, "same"), make_run_dir(runs, "same")
        self.assertEqual((first.name, second.name), ("same", "same_2"))

    def test_condition_target_trains(self):
        config = TrainConfig(**{**CONFIG.__dict__, "target": "condition", "epochs": 1})
        run_dir = self.run_training(config)
        self.assertEqual(json.loads((run_dir / "config.json").read_text())["classes"], ["bcc", "mel", "nv"])


if __name__ == "__main__":
    unittest.main()
