"""Create train/val/test split files, keeping each lesion or patient in one split.

External test datasets are never split: every image goes to test.csv.

Usage:
    python src/make_splits.py
    python src/make_splits.py --seed 42 --data-dir data --out-dir splits
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_NAMES = ("train", "val", "test")
ID_COLUMNS = ("image_id", "group_id")


@dataclass(frozen=True)
class Dataset:
    metadata: str  # relative to the data folder
    image_column: str
    group_column: str  # patient ID where the dataset has one, otherwise lesion ID
    label_column: str
    external_test: bool = False


DATASETS = {
    # HAM10000 has no patient ID, so lesion_id is the strictest grouping available.
    "ham10000": Dataset("ham10000/HAM10000_metadata.csv", "image_id", "lesion_id", "dx"),
    # File and column names are unverified until DDI is in data/. Loading fails if they differ.
    "ddi": Dataset("ddi/ddi_metadata.csv", "DDI_file", "DDI_file", "disease", external_test=True),
}


def load_manifest(dataset, data_dir):
    """Return one row per image with image_id, group_id and label columns."""
    raw = pd.read_csv(data_dir / dataset.metadata, dtype=str)
    columns = {
        "image_id": raw[dataset.image_column],
        "group_id": raw[dataset.group_column],
        "label": raw[dataset.label_column],
    }
    manifest = pd.DataFrame(columns).sort_values("image_id", ignore_index=True)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    missing = manifest[["image_id", "group_id", "label"]].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values would break grouping: {missing[missing > 0].to_dict()}")
    repeated = manifest["image_id"].duplicated().sum()
    if repeated:
        raise ValueError(f"{repeated} repeated image_id rows")


def assign_groups(manifest, seed):
    """Assign each group to a split, stratified by the group's most common label."""
    # sort_index makes ties resolve alphabetically, so the result does not depend on row order.
    group_labels = manifest.groupby("group_id")["label"].agg(lambda s: s.value_counts().sort_index().idxmax())
    rng = np.random.default_rng(seed)
    assignment = {}
    for label in sorted(group_labels.unique()):
        groups = np.array(sorted(group_labels.index[group_labels == label]))
        rng.shuffle(groups)
        n_test = round(len(groups) * TEST_FRACTION)
        n_val = round(len(groups) * VAL_FRACTION)
        assignment.update(dict.fromkeys(groups[:n_test], "test"))
        assignment.update(dict.fromkeys(groups[n_test : n_test + n_val], "val"))
        assignment.update(dict.fromkeys(groups[n_test + n_val :], "train"))
    return assignment


def make_splits(manifest, dataset, seed):
    """Return {split name: rows}. External test datasets only ever produce a test split."""
    if dataset.external_test:
        splits = {"test": manifest}
    else:
        assigned = manifest["group_id"].map(assign_groups(manifest, seed))
        splits = {name: manifest[assigned == name] for name in SPLIT_NAMES}
    check_no_overlap(splits)
    return splits


def check_no_overlap(splits):
    """Raise if any image or group ID appears in more than one split."""
    names = list(splits)
    for column in ID_COLUMNS:
        for position, first in enumerate(names):
            for second in names[position + 1 :]:
                shared = set(splits[first][column]) & set(splits[second][column])
                if shared:
                    raise ValueError(f"{len(shared)} {column} values in both {first} and {second}")


def save_splits(splits, name, dataset, seed, out_dir):
    dataset_dir = out_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        rows.to_csv(dataset_dir / f"{split}.csv", index=False)

    info = {
        "seed": seed,
        "group_column": dataset.group_column,
        "external_test": dataset.external_test,
        "images": {split: len(rows) for split, rows in splits.items()},
        "groups": {split: int(rows["group_id"].nunique()) for split, rows in splits.items()},
    }
    (dataset_dir / "split_info.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "splits")
    args = parser.parse_args()

    for name, dataset in DATASETS.items():
        if not (args.data_dir / dataset.metadata).is_file():
            print(f"{name}: skipped, {dataset.metadata} not found")
            continue
        manifest = load_manifest(dataset, args.data_dir)
        splits = make_splits(manifest, dataset, args.seed)
        save_splits(splits, name, dataset, args.seed, args.out_dir)
        counts = ", ".join(f"{split} {len(rows)}" for split, rows in splits.items())
        print(f"{name}: {counts} images, grouped by {dataset.group_column}")


if __name__ == "__main__":
    main()
