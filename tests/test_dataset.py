"""Tests for the image loader. Run from the repo root: python3 -m pytest tests"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.dataset import (
    DATA_DIR,
    SPLITS_DIR,
    TIERS,
    SkinDataset,
    eval_transform,
    train_transform,
)

TAXONOMY_COLUMNS = ["dataset", "source_label", "urgency_tier"]


class SyntheticData:
    """A tiny HAM10000-shaped dataset in a temporary folder."""

    def __init__(self, split_rows, taxonomy_rows, image_ids=None, mode="RGB"):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_dir, self.splits_dir = root / "data", root / "splits"
        self.taxonomy = root / "taxonomy.csv"

        part_1 = self.data_dir / "ham10000" / "HAM10000_images_part_1"
        part_2 = self.data_dir / "ham10000" / "HAM10000_images_part_2"
        part_1.mkdir(parents=True)
        part_2.mkdir(parents=True)
        ids = image_ids if image_ids is not None else [r[0] for r in split_rows]
        for n, image_id in enumerate(ids):
            folder = part_1 if n % 2 == 0 else part_2  # spread across both folders
            Image.new(mode, (60, 45)).save(folder / f"{image_id}.jpg")

        (self.splits_dir / "ham10000").mkdir(parents=True)
        pd.DataFrame(split_rows, columns=["image_id", "group_id", "label"]).to_csv(
            self.splits_dir / "ham10000" / "train.csv", index=False)
        pd.DataFrame(taxonomy_rows, columns=TAXONOMY_COLUMNS).to_csv(self.taxonomy, index=False)

    def load(self, target, transform=None):
        return SkinDataset("ham10000", "train", target=target, transform=transform,
                           data_dir=self.data_dir, splits_dir=self.splits_dir,
                           taxonomy_path=self.taxonomy)

    def close(self):
        self.tmp.cleanup()


SPLIT = [("img_a", "g1", "mel"), ("img_b", "g2", "nv"), ("img_c", "g2", "nv"), ("img_d", "g3", "bcc")]
TAXONOMY = [("ham10000", "nv", "Routine"), ("ham10000", "mel", "Urgent"), ("ham10000", "bcc", "Soon"),
            ("ddi", "melanoma", "Routine")]  # another dataset's row must be ignored


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.data = SyntheticData(SPLIT, TAXONOMY)

    def tearDown(self):
        self.data.close()

    def test_condition_classes_are_sorted_taxonomy_labels(self):
        ds = self.data.load("condition")
        self.assertEqual(ds.classes, ["bcc", "mel", "nv"])
        self.assertEqual(ds.labels, [1, 2, 2, 0])

    def test_tier_target_uses_taxonomy_and_fixed_order(self):
        ds = self.data.load("tier")
        self.assertEqual(ds.classes, list(TIERS))
        self.assertEqual([ds.classes[i] for i in ds.labels], ["Urgent", "Routine", "Routine", "Soon"])

    def test_class_counts(self):
        self.assertEqual(self.data.load("tier").class_counts().tolist(), [1, 1, 2])

    def test_images_found_in_both_folders(self):
        ds = self.data.load("condition")
        self.assertEqual({p.parent.name for p in ds.paths},
                         {"HAM10000_images_part_1", "HAM10000_images_part_2"})

    def test_item_shape_and_type(self):
        for transform in (train_transform(), eval_transform()):
            image, label = self.data.load("tier", transform)[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(image.dtype, torch.float32)
            self.assertIsInstance(label, int)

    def test_eval_transform_is_deterministic(self):
        ds = self.data.load("tier", eval_transform())
        self.assertTrue(torch.equal(ds[0][0], ds[0][0]))

    def test_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            self.data.load("diagnosis")


class LoaderErrorTests(unittest.TestCase):
    def test_label_without_tier_is_rejected(self):
        data = SyntheticData(SPLIT, [r if r[1] != "bcc" else ("ham10000", "bcc", "") for r in TAXONOMY])
        try:
            data.load("condition")  # conditions do not need tiers
            with self.assertRaisesRegex(ValueError, "bcc"):
                data.load("tier")
        finally:
            data.close()

    def test_label_missing_from_taxonomy_is_rejected(self):
        data = SyntheticData(SPLIT, [r for r in TAXONOMY if r[1] != "bcc"])
        try:
            with self.assertRaisesRegex(ValueError, "bcc"):
                data.load("condition")
        finally:
            data.close()

    def test_missing_image_is_rejected(self):
        data = SyntheticData(SPLIT, TAXONOMY, image_ids=["img_a", "img_b", "img_c"])
        try:
            with self.assertRaises(FileNotFoundError):
                data.load("condition")
        finally:
            data.close()

    def test_greyscale_image_becomes_rgb(self):
        data = SyntheticData(SPLIT, TAXONOMY, mode="L")
        try:
            image, _ = data.load("tier", eval_transform())[0]
            self.assertEqual(image.shape[0], 3)
        finally:
            data.close()


@unittest.skipUnless((DATA_DIR / "ham10000").is_dir() and (SPLITS_DIR / "ham10000").is_dir(),
                     "HAM10000 data not available")
class RealHam10000Tests(unittest.TestCase):
    def test_every_split_loads_with_both_targets(self):
        expected = {"train": 7037, "val": 1491, "test": 1487}
        for split, size in expected.items():
            for target in ("condition", "tier"):
                with self.subTest(split=split, target=target):
                    ds = SkinDataset("ham10000", split, target=target, transform=eval_transform())
                    self.assertEqual(len(ds), size)
                    self.assertEqual(ds[0][0].shape, (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
