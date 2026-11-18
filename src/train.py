"""Train one model on one dataset and target, logging its configuration and metrics.

Each run gets its own folder under runs/ containing:
    config.json   every setting, the git commit, package versions, classes and class weights
    metrics.csv   one row per epoch: losses, accuracy, balanced accuracy, learning rate, time
    best.pt       weights from the epoch with the highest validation balanced accuracy
    summary.json  best epoch and its validation metrics

Usage:
    python src/train.py --target tier
    python src/train.py --target condition --arch resnet18 --epochs 10
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import balanced_accuracy_score

if __package__ in (None, ""):  # allow "python src/train.py" as well as "python -m src.train"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (  # noqa: E402
    DATA_DIR,
    SPLITS_DIR,
    TARGETS,
    TAXONOMY_PATH,
    SkinDataset,
    eval_transform,
    train_transform,
)
from src.model import ARCHITECTURES, build_model, class_weights, weighted_loss  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class TrainConfig:
    dataset: str = "ham10000"
    target: str = "tier"
    arch: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.0
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5  # stop after this many epochs without a better validation score
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def seed_worker(worker_id):
    """Give each data-loading worker its own seed derived from the main seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_state():
    """Commit hash and whether there were uncommitted changes, so a run can be traced to its code."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(git("status", "--porcelain"))}
    except FileNotFoundError:
        return {"commit": None, "uncommitted_changes": None}


def make_loader(dataset, config, shuffle):
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, labels_all, preds_all = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total += loss_fn(logits, labels).item() * len(labels)
        labels_all.append(labels.cpu())
        preds_all.append(logits.argmax(dim=1).cpu())
    y_true, y_pred = torch.cat(labels_all).numpy(), torch.cat(preds_all).numpy()
    return {
        "loss": total / len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_name(config):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{config.dataset}_{config.target}_{config.arch}_seed{config.seed}"


def make_run_dir(runs_dir, name):
    """Create a new run folder, adding _2, _3, ... if runs started in the same second."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate, n = runs_dir / name, 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1
            candidate = runs_dir / f"{name}_{n}"


def train(config,data_dir=DATA_DIR, splits_dir=SPLITS_DIR, taxonomy_path=TAXONOMY_PATH, runs_dir=RUNS_DIR):
    """Train one model and return the path of its run folder."""
    set_seed(config.seed)
    device = pick_device(config.device)
    paths = {"data_dir": data_dir, "splits_dir": splits_dir, "taxonomy_path": taxonomy_path}

    train_set = SkinDataset(config.dataset, "train", config.target, train_transform(), **paths)
    val_set = SkinDataset(config.dataset, "val", config.target, eval_transform(), **paths)
    train_loader = make_loader(train_set, config, shuffle=True)
    val_loader = make_loader(val_set, config, shuffle=False)

    model = build_model(len(train_set.classes), config.arch, config.pretrained, config.dropout).to(device)
    loss_fn = weighted_loss(train_set.class_counts(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    run_dir = make_run_dir(runs_dir, run_name(config))
    (run_dir / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "device": str(device),
        "classes": train_set.classes,
        "train_class_counts": train_set.class_counts().tolist(),
        "class_weights": [round(w, 6) for w in class_weights(train_set.class_counts()).tolist()],
        "split_sizes": {"train": len(train_set), "val": len(val_set)},
        "git": git_state(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
        "started": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")

    fields = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy", "lr", "seconds"]
    best = {"epoch": 0, "val_balanced_accuracy": -1.0}
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val = evaluate(model, val_loader, loss_fn, device)
            scheduler.step()

            row = {"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val["loss"], 6),
                   "val_accuracy": round(val["accuracy"], 6),
                   "val_balanced_accuracy": round(val["balanced_accuracy"], 6),
                   "lr": lr, "seconds": round(time.time() - start, 1)}
            writer.writerow(row)
            f.flush()
            print(f"epoch {epoch}/{config.epochs}  train loss {train_loss:.4f}  val loss {val['loss']:.4f}  "
                  f"val bal acc {val['balanced_accuracy']:.4f}  ({row['seconds']}s)", flush=True)

            if val["balanced_accuracy"] > best["val_balanced_accuracy"]:
                best = {"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}}
                torch.save({"model_state": model.state_dict(), "classes": train_set.classes,
                            "config": asdict(config), "epoch": epoch}, run_dir / "best.pt")
            elif epoch - best["epoch"] >= config.patience:
                print(f"Stopping early: no improvement for {config.patience} epochs.")
                break

    summary = {"best_epoch": best["epoch"], "epochs_run": epoch, **{k: v for k, v in best.items() if k != "epoch"},
               "finished": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Best epoch {best['epoch']}, val balanced accuracy {best['val_balanced_accuracy']:.4f}. Saved to {run_dir}")
    return run_dir


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=defaults.dataset)
    parser.add_argument("--target", choices=TARGETS, default=defaults.target)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default=defaults.arch)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    for name in ("dropout", "lr", "weight_decay"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    for name in ("epochs", "batch_size", "patience", "seed", "num_workers"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, mps or cuda")
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    train(parse_args())
