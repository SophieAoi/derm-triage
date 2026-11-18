"""Evaluate a trained run at the urgency-tier level.

Works for both kinds of model, so they can be compared directly (RQ1):
    tier models       predict Urgent / Soon / Routine directly
    condition models  predict a condition; each condition's probability is added to
                      its tier ("summed"), and the top condition's tier is also
                      reported on its own ("top-condition")

Writes to <run>/eval_<split>_<mapping>/:
    predictions.csv  one row per image: true labels, predictions, tier probabilities
    metrics.json     accuracy, balanced accuracy, per-tier sensitivity and specificity,
                     under/over-triage rates and confusion matrices

Usage:
    python src/evaluate.py runs/<run folder>                  # validation split
    python src/evaluate.py runs/<run folder> --split test     # held-out test split
    python src/evaluate.py runs/<run folder> --mapping cautious
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import (  # noqa: E402
    DATA_DIR,
    SPLITS_DIR,
    TAXONOMY_PATH,
    TIERS,
    SkinDataset,
    eval_transform,
    load_tier_map,
)
from src.model import build_model  # noqa: E402
from src.train import make_loader, pick_device, TrainConfig  # noqa: E402

# Alternative tier mappings for the sensitivity analyses in DOCUMENTATION.md.
# Each one overrides the taxonomy's tier for the listed labels.
MAPPINGS = {
    "default": {},
    "cautious": {"bcc": "Urgent", "akiec": "Urgent"},
    "relaxed": {"akiec": "Routine"},
}
# Splits a run may be evaluated on. DDI is not listed: it is never loaded here.
SPLITS = ("val", "test")


def tier_map_for(dataset, mapping, taxonomy_path=TAXONOMY_PATH):
    tier_map = load_tier_map(dataset, taxonomy_path)
    return {**tier_map, **{k: v for k, v in MAPPINGS[mapping].items() if k in tier_map}}


def condition_to_tier_probs(condition_probs, conditions, tier_map):
    """Add up condition probabilities into tier probabilities. Columns follow TIERS."""
    to_tier = np.zeros((len(conditions), len(TIERS)))
    for i, condition in enumerate(conditions):
        to_tier[i, TIERS.index(tier_map[condition])] = 1.0
    return condition_probs @ to_tier


def confusion(y_true, y_pred, n):
    """Rows are true classes, columns are predicted classes."""
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def tier_metrics(y_true, y_pred):
    """Tier-level metrics. Labels are indices into TIERS, ordered most to least urgent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    matrix = confusion(y_true, y_pred, len(TIERS))
    per_tier = {}
    for i, tier in enumerate(TIERS):
        tp = matrix[i, i]
        fn = matrix[i].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_tier[tier] = {
            "support": int(tp + fn),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
        }
    sensitivities = [m["sensitivity"] for m in per_tier.values() if m["sensitivity"] is not None]
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(sensitivities)),
        # A higher index is less urgent, so predicting a higher index than the truth is under-triage.
        "undertriage_rate": float((y_pred > y_true).mean()),
        "overtriage_rate": float((y_pred < y_true).mean()),
        "per_tier": per_tier,
        "confusion_matrix": {"rows_true_cols_predicted": list(TIERS), "matrix": matrix.tolist()},
    }


@torch.no_grad()
def predict(model, dataset, device, batch_size=64, num_workers=4):
    loader = make_loader(dataset, TrainConfig(batch_size=batch_size, num_workers=num_workers), shuffle=False)
    model.eval()
    probs = [torch.softmax(model(images.to(device)), dim=1).cpu() for images, _ in loader]
    return torch.cat(probs).numpy()


def evaluate_run(run_dir, split="val", mapping="default", data_dir=DATA_DIR, splits_dir=SPLITS_DIR,
                 taxonomy_path=TAXONOMY_PATH, device="auto", num_workers=4):
    """Evaluate a run's best checkpoint and return (metrics, output folder)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {sorted(MAPPINGS)}, not {mapping!r}")
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["target"] == "tier" and mapping != "default":
        raise ValueError("A tier model was trained on the default mapping; another mapping needs retraining.")

    dataset = SkinDataset(config["dataset"], split, config["target"], eval_transform(),
                          data_dir=data_dir, splits_dir=splits_dir, taxonomy_path=taxonomy_path)
    if dataset.classes != checkpoint["classes"]:
        raise ValueError(f"Class mismatch: checkpoint {checkpoint['classes']}, data {dataset.classes}")

    device = pick_device(device)
    model = build_model(len(dataset.classes), config["arch"], pretrained=False, dropout=config["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    probs = predict(model.to(device), dataset, device, num_workers=num_workers)

    tier_map = tier_map_for(config["dataset"], mapping, taxonomy_path)
    true_tier = np.array([TIERS.index(tier_map[c]) for c in dataset.conditions])
    predictions = pd.DataFrame({"image_id": dataset.image_ids, "true_condition": dataset.conditions,
                                "true_tier": [TIERS[i] for i in true_tier]})

    if config["target"] == "tier":
        tier_probs = probs
        metrics = {"tier": tier_metrics(true_tier, tier_probs.argmax(1))}
    else:
        tier_probs = condition_to_tier_probs(probs, dataset.classes, tier_map)
        top_condition = probs.argmax(1)
        top_condition_tier = np.array([TIERS.index(tier_map[dataset.classes[i]]) for i in top_condition])
        predictions["predicted_condition"] = [dataset.classes[i] for i in top_condition]
        predictions["top_condition_tier"] = [TIERS[i] for i in top_condition_tier]
        cond_true = np.array(dataset.labels)
        metrics = {
            "tier": tier_metrics(true_tier, tier_probs.argmax(1)),
            "tier_from_top_condition": tier_metrics(true_tier, top_condition_tier),
            "condition": {
                "accuracy": float((cond_true == top_condition).mean()),
                "per_condition_sensitivity": {
                    c: float((top_condition[cond_true == i] == i).mean()) if (cond_true == i).any() else None
                    for i, c in enumerate(dataset.classes)},
                "confusion_matrix": {"rows_true_cols_predicted": dataset.classes,
                                     "matrix": confusion(cond_true, top_condition, len(dataset.classes)).tolist()},
            },
        }
    predictions["predicted_tier"] = [TIERS[i] for i in tier_probs.argmax(1)]
    for i, tier in enumerate(TIERS):
        predictions[f"p_{tier.lower()}"] = tier_probs[:, i].round(6)

    metrics = {"run": run_dir.name, "dataset": config["dataset"], "target": config["target"],
               "split": split, "mapping": mapping, "best_epoch": checkpoint["epoch"], **metrics}
    out_dir = run_dir / f"eval_{split}_{mapping}"
    out_dir.mkdir(exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, out_dir


def format_report(metrics):
    lines = [f"{metrics['run']} | {metrics['split']} split | {metrics['mapping']} mapping"]
    sections = [("tier", "Tiers")]
    if "tier_from_top_condition" in metrics:
        sections = [("tier", "Tiers (summed condition probabilities)"),
                    ("tier_from_top_condition", "Tiers (tier of top condition)")]
    for key, title in sections:
        m = metrics[key]
        lines += ["", f"{title}: accuracy {m['accuracy']:.3f}, balanced accuracy {m['balanced_accuracy']:.3f}, "
                      f"under-triage {m['undertriage_rate']:.3f}, over-triage {m['overtriage_rate']:.3f}",
                  f"  {'tier':<8}{'n':>6}{'sens':>8}{'spec':>8}"]
        for tier, t in m["per_tier"].items():
            fmt = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"  # noqa: E731
            lines.append(f"  {tier:<8}{t['support']:>6}{fmt(t['sensitivity'])}{fmt(t['specificity'])}")
        lines.append(f"  confusion (rows true, cols predicted {', '.join(TIERS)}):")
        lines += [f"  {tier:<8}" + "".join(f"{v:>7}" for v in row)
                  for tier, row in zip(TIERS, m["confusion_matrix"]["matrix"])]
    if "condition" in metrics:
        c = metrics["condition"]
        lines += ["", f"Conditions: accuracy {c['accuracy']:.3f}; sensitivity " +
                  ", ".join(f"{k} {v:.3f}" for k, v in c["per_condition_sensitivity"].items() if v is not None)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--mapping", choices=sorted(MAPPINGS), default="default")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metrics, out_dir = evaluate_run(args.run_dir, args.split, args.mapping, device=args.device)
    print(format_report(metrics))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
