"""Image loading and augmentation for training and evaluation.

Reads the split files written by make_splits.py (image_id, group_id, label) and
returns (image tensor, class index) pairs. Two targets are supported:
    "condition": the dataset's own diagnosis label (e.g. mel, nv)
    "tier":      the urgency tier for that label, looked up in taxonomy/taxonomy.csv

Example:
    train = SkinDataset("ham10000", "train", target="tier", transform=train_transform())
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = REPO_ROOT / "splits"
TAXONOMY_PATH = REPO_ROOT / "taxonomy" / "taxonomy.csv"

# Folders holding each dataset's images, relative to the data folder.
IMAGE_DIRS = {
    "ham10000": ["ham10000/HAM10000_images_part_1", "ham10000/HAM10000_images_part_2"],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Fixed order, so class index 0 always means Urgent.
TIERS = ("Urgent", "Soon", "Routine")
TARGETS = ("condition", "tier")

IMAGE_SIZE = 224
# ImageNet statistics, matching the pretrained weights the model will start from.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size=IMAGE_SIZE):
    """Random crops, flips, rotations and mild lighting changes.

    Hue is deliberately left unchanged: lesion colour is diagnostic, and shifting
    hue would also change apparent skin tone.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size=IMAGE_SIZE):
    """Deterministic resize and centre crop for validation and testing."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def index_images(dataset, data_dir=DATA_DIR):
    """Return {image_id: path} for every image file in the dataset's folders."""
    if dataset not in IMAGE_DIRS:
        raise ValueError(f"No image folders configured for {dataset!r}")
    index = {}
    for folder in IMAGE_DIRS[dataset]:
        for path in sorted((data_dir / folder).iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.stem in index:
                raise ValueError(f"Image {path.stem} found in two places: {index[path.stem]} and {path}")
            index[path.stem] = path
    return index


def load_tier_map(dataset, taxonomy_path=TAXONOMY_PATH):
    """Return {source_label: urgency tier} for one dataset's rows in the taxonomy."""
    taxonomy = pd.read_csv(taxonomy_path, dtype=str).fillna("")
    rows = taxonomy[taxonomy["dataset"] == dataset]
    unknown = set(rows["urgency_tier"]) - set(TIERS) - {""}
    if unknown:
        raise ValueError(f"Unknown urgency tiers in taxonomy: {sorted(unknown)}")
    return dict(zip(rows["source_label"], rows["urgency_tier"]))


def condition_classes(dataset, taxonomy_path=TAXONOMY_PATH):
    """Sorted list of the dataset's condition labels, so class order never depends on a split."""
    taxonomy = pd.read_csv(taxonomy_path, dtype=str)
    return sorted(taxonomy.loc[taxonomy["dataset"] == dataset, "source_label"])


class SkinDataset(Dataset):
    def __init__(self, dataset, split, target="condition", transform=None,
                 data_dir=DATA_DIR, splits_dir=SPLITS_DIR, taxonomy_path=TAXONOMY_PATH):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        self.dataset, self.split, self.target, self.transform = dataset, split, target, transform

        rows = pd.read_csv(splits_dir / dataset / f"{split}.csv", dtype=str)
        self.image_ids = rows["image_id"].tolist()
        self.conditions = rows["label"].tolist()

        cond_classes = condition_classes(dataset, taxonomy_path)
        unlisted = set(self.conditions) - set(cond_classes)
        if unlisted:
            raise ValueError(f"Labels in {split} split missing from taxonomy: {sorted(unlisted)}")

        if target == "condition":
            self.classes = cond_classes
            names = self.conditions
        else:
            tier_map = load_tier_map(dataset, taxonomy_path)
            untiered = sorted({c for c in self.conditions if not tier_map.get(c)})
            if untiered:
                raise ValueError(f"No urgency tier in taxonomy for: {untiered}")
            self.classes = list(TIERS)
            names = [tier_map[c] for c in self.conditions]
        class_index = {name: i for i, name in enumerate(self.classes)}
        self.labels = [class_index[name] for name in names]

        index = index_images(dataset, data_dir)
        missing = [i for i in self.image_ids if i not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images in {split} split not found, e.g. {missing[:3]}")
        self.paths = [index[i] for i in self.image_ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            image = img.convert("RGB")  # also handles greyscale and RGBA files
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]

    def class_counts(self):
        """Number of images per class, in class order (for loss weighting)."""
        return torch.bincount(torch.tensor(self.labels), minlength=len(self.classes))
