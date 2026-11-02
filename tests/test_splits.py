"""Tests for the split code. Run from the repo root: python3 -m unittest discover -s tests"""

import unittest

import pandas as pd

from src.make_splits import (
    DATASETS,
    REPO_ROOT,
    Dataset,
    check_no_overlap,
    make_splits,
    validate_manifest,
)

SPLITS_DIR = REPO_ROOT / "splits"
HOSPITAL = Dataset("unused.csv", "image_id", "group_id", "label")
EXTERNAL = Dataset("unused.csv", "image_id", "group_id", "label", external_test=True)


def synthetic_manifest():
    """200 groups with 1 to 3 images each, across three unevenly sized classes."""
    rows = []
    for group in range(200):
        label = "a" if group % 10 < 7 else "b" if group % 10 < 9 else "c"
        for image in range(group % 3 + 1):
            rows.append({"image_id": f"img_{group}_{image}", "group_id": f"group_{group}", "label": label})
    return pd.DataFrame(rows)


def shared_ids(splits, column):
    """Return IDs found in more than one split, without using the code under test."""
    seen = pd.concat(rows[[column]].drop_duplicates() for rows in splits.values())
    return set(seen.loc[seen[column].duplicated(), column])


class SplitLogicTests(unittest.TestCase):
    def test_no_id_appears_in_more_than_one_split(self):
        manifest = synthetic_manifest()
        for seed in range(20):
            splits = make_splits(manifest, HOSPITAL, seed)
            self.assertEqual(shared_ids(splits, "group_id"), set())
            self.assertEqual(shared_ids(splits, "image_id"), set())

    def test_every_image_is_kept_once(self):
        manifest = synthetic_manifest()
        splits = make_splits(manifest, HOSPITAL, 0)
        kept = pd.concat(splits.values())
        self.assertEqual(sorted(kept["image_id"]), sorted(manifest["image_id"]))

    def test_same_seed_gives_same_split(self):
        manifest = synthetic_manifest()
        first = make_splits(manifest, HOSPITAL, 7)
        second = make_splits(manifest, HOSPITAL, 7)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_row_order_does_not_change_the_split(self):
        manifest = synthetic_manifest()
        shuffled = manifest.sample(frac=1, random_state=1)
        first = make_splits(manifest, HOSPITAL, 7)
        second = make_splits(shuffled, HOSPITAL, 7)
        for name in first:
            self.assertEqual(set(first[name]["image_id"]), set(second[name]["image_id"]))

    def test_different_seed_gives_different_split(self):
        manifest = synthetic_manifest()
        first = make_splits(manifest, HOSPITAL, 1)
        second = make_splits(manifest, HOSPITAL, 2)
        self.assertNotEqual(set(first["test"]["group_id"]), set(second["test"]["group_id"]))

    def test_every_class_reaches_every_split(self):
        splits = make_splits(synthetic_manifest(), HOSPITAL, 0)
        for name, rows in splits.items():
            self.assertEqual(set(rows["label"]), {"a", "b", "c"}, name)

    def test_group_with_mixed_labels_stays_together(self):
        manifest = synthetic_manifest()
        extra = pd.DataFrame([{"image_id": "mixed", "group_id": "group_0", "label": "b"}])
        splits = make_splits(pd.concat([manifest, extra], ignore_index=True), HOSPITAL, 0)
        self.assertEqual(shared_ids(splits, "group_id"), set())

    def test_external_dataset_is_test_only(self):
        manifest = synthetic_manifest()
        splits = make_splits(manifest, EXTERNAL, 0)
        self.assertEqual(list(splits), ["test"])
        self.assertEqual(len(splits["test"]), len(manifest))

    def test_ddi_is_configured_as_external(self):
        self.assertTrue(DATASETS["ddi"].external_test)

    def test_overlap_check_catches_a_shared_group(self):
        manifest = synthetic_manifest()
        splits = make_splits(manifest, HOSPITAL, 0)
        leaked = splits["test"].iloc[[0]].assign(image_id="leaked_image")
        splits["train"] = pd.concat([splits["train"], leaked])
        with self.assertRaisesRegex(ValueError, "group_id"):
            check_no_overlap(splits)

    def test_missing_group_id_is_rejected(self):
        manifest = synthetic_manifest()
        manifest.loc[0, "group_id"] = None
        with self.assertRaisesRegex(ValueError, "group_id"):
            validate_manifest(manifest)

    def test_repeated_image_id_is_rejected(self):
        manifest = synthetic_manifest()
        manifest.loc[1, "image_id"] = manifest.loc[0, "image_id"]
        with self.assertRaisesRegex(ValueError, "repeated"):
            validate_manifest(manifest)


class SavedSplitTests(unittest.TestCase):
    """Checks the files in splits/, which are what training will read."""

    def saved_datasets(self):
        found = [name for name in DATASETS if (SPLITS_DIR / name).is_dir()]
        if not found:
            self.skipTest("no saved splits yet, run src/make_splits.py")
        return found

    def load(self, name):
        files = sorted((SPLITS_DIR / name).glob("*.csv"))
        return {path.stem: pd.read_csv(path, dtype=str) for path in files}

    def test_no_id_appears_in_more_than_one_saved_split(self):
        for name in self.saved_datasets():
            splits = self.load(name)
            with self.subTest(dataset=name):
                self.assertEqual(shared_ids(splits, "group_id"), set())
                self.assertEqual(shared_ids(splits, "image_id"), set())

    def test_saved_files_match_the_dataset_role(self):
        for name in self.saved_datasets():
            expected = {"test"} if DATASETS[name].external_test else {"train", "val", "test"}
            with self.subTest(dataset=name):
                self.assertEqual(set(self.load(name)), expected)


if __name__ == "__main__":
    unittest.main()
