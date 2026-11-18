"""Softmax-based uncertainty, escalation, risk-coverage and calibration on the validation split.

Uses existing checkpoints only: nothing is retrained, and the test split and DDI are
never loaded. Each run gets a new uncertainty_val/ subfolder; nothing else in the run
folder is changed. Figures that compare runs go in a separate comparison folder.

Two uncertainty scores are used, each from the model's own softmax (3 tiers for a tier
model, 7 conditions for a condition model):
    entropy          -sum p log p, in nats
    1 - max softmax  one minus the highest class probability

Escalation: a case whose uncertainty is above the threshold is sent to the Urgent tier
for human review, whatever tier was predicted.

Usage:
    python src/uncertainty.py runs/<tier run> runs/<condition run>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import DATA_DIR, SPLITS_DIR, TAXONOMY_PATH, TIERS, SkinDataset, eval_transform  # noqa: E402
from src.evaluate import condition_to_tier_probs, tier_map_for  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import TrainConfig, make_loader, pick_device  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT = "val"  # the only split this script ever loads
URGENT, ROUTINE = TIERS.index("Urgent"), TIERS.index("Routine")
N_THRESHOLDS = 100
ECE_BINS = 15
SCORES = {"entropy": "entropy", "one_minus_max_softmax": "1 - max softmax"}
SAME_SPLIT_WARNING = (
    "Temperature fitted and evaluated on the same validation split, so the post-scaling "
    "calibration is optimistic. The final temperature and escalation threshold must be chosen "
    "on validation and then applied unchanged to the test split."
)


# ---------------------------------------------------------------- core functions

def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(probs):
    """Shannon entropy of each row, in nats. 0 for one-hot, log(K) for uniform over K classes."""
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=1)


def one_minus_max_softmax(probs):
    return 1.0 - np.asarray(probs).max(axis=1)


def max_score(score, n_classes):
    """Largest possible value of an uncertainty score (reached by a uniform distribution)."""
    return np.log(n_classes) if score == "entropy" else 1.0 - 1.0 / n_classes


def escalate(pred_tier, uncertainty, threshold):
    """Final tiers after sending every case with uncertainty above threshold to Urgent."""
    escalated = np.asarray(uncertainty) > threshold
    return np.where(escalated, URGENT, pred_tier), escalated


def escalation_metrics(true_tier, pred_tier, uncertainty, threshold):
    true_tier, pred_tier = np.asarray(true_tier), np.asarray(pred_tier)
    final, escalated = escalate(pred_tier, uncertainty, threshold)
    is_urgent = true_tier == URGENT
    n_urgent, n_other = int(is_urgent.sum()), int((~is_urgent).sum())
    missed_before = is_urgent & (pred_tier != URGENT)
    return {
        "threshold": float(threshold),
        "escalated": int(escalated.sum()),
        "escalation_rate": float(escalated.mean()),
        "urgent_sensitivity": float((final[is_urgent] == URGENT).mean()) if n_urgent else np.nan,
        # Rates use the group they describe as the denominator (see the report notes).
        "undertriage_urgent_to_routine": int((is_urgent & (final == ROUTINE)).sum()),
        "undertriage_rate": float((is_urgent & (final == ROUTINE)).sum() / n_urgent) if n_urgent else np.nan,
        "overtriage_to_urgent": int((~is_urgent & (final == URGENT)).sum()),
        "overtriage_rate": float((~is_urgent & (final == URGENT)).sum() / n_other) if n_other else np.nan,
        "missed_urgent_caught": int((missed_before & escalated).sum()),
        "missed_urgent_still_missed": int((missed_before & ~escalated).sum()),
        # Correct non-Urgent predictions that escalation moved up to Urgent.
        "correct_sent_up_needlessly": int((escalated & (pred_tier == true_tier) & ~is_urgent).sum()),
    }


def escalation_sweep(true_tier, pred_tier, uncertainty, n_classes, score, n_thresholds=N_THRESHOLDS):
    """Metrics at n_thresholds evenly spaced thresholds from 0 to the score's maximum."""
    thresholds = np.linspace(0.0, max_score(score, n_classes), n_thresholds)
    return pd.DataFrame([escalation_metrics(true_tier, pred_tier, uncertainty, t) for t in thresholds])


def risk_coverage(errors, uncertainty):
    """Keep the k most certain cases for k = 1..N. Returns (coverage, risk, AURC).

    Risk is the error rate on the kept cases; AURC is the mean risk over all coverage
    levels (lower is better). Ties in uncertainty keep the original order.
    """
    errors = np.asarray(errors, dtype=np.float64)
    order = np.argsort(uncertainty, kind="stable")
    kept = np.arange(1, len(errors) + 1)
    risk = np.cumsum(errors[order]) / kept
    return kept / len(errors), risk, float(risk.mean())


def error_on_most_uncertain(errors, uncertainty, fraction):
    """Error rate on the most uncertain `fraction` of cases."""
    errors = np.asarray(errors, dtype=np.float64)
    n = max(1, int(round(fraction * len(errors))))
    order = np.argsort(uncertainty, kind="stable")
    return float(errors[order[-n:]].mean())


def calibration_bins(probs, labels, n_bins=ECE_BINS):
    """Equal-width confidence bins on (0, 1]. Returns a table with count, confidence and accuracy."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_of == b
        rows.append({"bin_lower": edges[b], "bin_upper": edges[b + 1], "count": int(mask.sum()),
                     "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                     "accuracy": float(correct[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(probs, labels, n_bins=ECE_BINS):
    """Weighted mean gap between confidence and accuracy across equal-width bins."""
    bins = calibration_bins(probs, labels, n_bins)
    filled = bins[bins["count"] > 0]
    gaps = (filled["accuracy"] - filled["mean_confidence"]).abs()
    return float((gaps * filled["count"]).sum() / filled["count"].sum())


def fit_temperature(logits, labels):
    """Single temperature T > 0 minimising negative log-likelihood of softmax(logits / T)."""
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # optimise log T so T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def nll(probs, labels):
    p = np.asarray(probs)[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


# ---------------------------------------------------------------- running on a checkpoint

@torch.no_grad()
def model_logits(run_dir, device="auto", num_workers=4, data_dir=DATA_DIR, splits_dir=SPLITS_DIR,
                 taxonomy_path=TAXONOMY_PATH):
    """Validation logits from a run's best checkpoint. Returns (logits, dataset, config)."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    dataset = SkinDataset(config["dataset"], SPLIT, config["target"], eval_transform(),
                          data_dir=data_dir, splits_dir=splits_dir, taxonomy_path=taxonomy_path)
    if dataset.classes != checkpoint["classes"]:
        raise ValueError(f"Class mismatch: checkpoint {checkpoint['classes']}, data {dataset.classes}")
    device = pick_device(device)
    model = build_model(len(dataset.classes), config["arch"], pretrained=False, dropout=config["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    loader = make_loader(dataset, TrainConfig(batch_size=64, num_workers=num_workers), shuffle=False)
    logits = torch.cat([model(images.to(device)).cpu() for images, _ in loader]).double().numpy()
    return logits, dataset, config


def analyse_run(run_dir, taxonomy_path=TAXONOMY_PATH, **kwargs):
    """Run every analysis for one model. Writes run_dir/uncertainty_val/ and returns the results."""
    run_dir = Path(run_dir)
    logits, dataset, config = model_logits(run_dir, taxonomy_path=taxonomy_path, **kwargs)
    tier_map = tier_map_for(config["dataset"], "default", taxonomy_path)
    true_tier = np.array([TIERS.index(tier_map[c]) for c in dataset.conditions])
    own_labels = np.array(dataset.labels)  # tiers or conditions, matching the softmax
    probs = softmax(logits)
    is_condition = config["target"] == "condition"
    tier_probs = condition_to_tier_probs(probs, dataset.classes, tier_map) if is_condition else probs
    pred_tier = tier_probs.argmax(axis=1)
    errors = pred_tier != true_tier
    scores = {"entropy": entropy(probs), "one_minus_max_softmax": one_minus_max_softmax(probs)}

    # 1. Per-image outputs
    table = pd.DataFrame({"image_id": dataset.image_ids, "true_condition": dataset.conditions,
                          "true_tier": [TIERS[i] for i in true_tier],
                          "predicted_tier": [TIERS[i] for i in pred_tier]})
    if is_condition:
        table["predicted_condition"] = [dataset.classes[i] for i in probs.argmax(axis=1)]
        for i, c in enumerate(dataset.classes):
            table[f"p_{c}"] = probs[:, i]
    for i, tier in enumerate(TIERS):
        table[f"p_{tier.lower()}"] = tier_probs[:, i]
    table["max_softmax"] = probs.max(axis=1)
    table["entropy"] = scores["entropy"]
    if is_condition:
        table["tier_max_prob"] = tier_probs.max(axis=1)

    out_dir = run_dir / "uncertainty_val"
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "predictions.csv", index=False, float_format="%.6g")

    # 2. Escalation sweeps
    sweeps = {}
    for score, values in scores.items():
        sweep = escalation_sweep(true_tier, pred_tier, values, len(dataset.classes), score)
        sweep.insert(0, "score", score)
        sweeps[score] = sweep
    pd.concat(sweeps.values()).to_csv(out_dir / "escalation_sweep.csv", index=False, float_format="%.6g")

    # 3. Risk-coverage
    risk = {}
    curves = []
    for score, values in scores.items():
        coverage, r, aurc = risk_coverage(errors, values)
        curves.append(pd.DataFrame({"score": score, "coverage": coverage, "risk": r}))
        risk[score] = {"aurc": aurc,
                       "error_most_uncertain_10pct": error_on_most_uncertain(errors, values, 0.10),
                       "error_most_uncertain_20pct": error_on_most_uncertain(errors, values, 0.20)}
    pd.concat(curves).to_csv(out_dir / "risk_coverage.csv", index=False, float_format="%.6g")

    # 4. Calibration: temperature fitted on the model's own outputs (tiers or conditions)
    temperature = fit_temperature(logits, own_labels)
    scaled = softmax(logits, temperature)
    scaled_tier = condition_to_tier_probs(scaled, dataset.classes, tier_map) if is_condition else scaled
    calibration = {
        "temperature": temperature,
        "warning": SAME_SPLIT_WARNING,
        "tier_level": {"ece_before": expected_calibration_error(tier_probs, true_tier),
                       "ece_after": expected_calibration_error(scaled_tier, true_tier),
                       "nll_before": nll(tier_probs, true_tier), "nll_after": nll(scaled_tier, true_tier)},
    }
    if is_condition:
        calibration["condition_level"] = {
            "ece_before": expected_calibration_error(probs, own_labels),
            "ece_after": expected_calibration_error(scaled, own_labels),
            "nll_before": nll(probs, own_labels), "nll_after": nll(scaled, own_labels)}
    reliability = {"before": calibration_bins(tier_probs, true_tier), "after": calibration_bins(scaled_tier, true_tier)}
    pd.concat([df.assign(stage=stage) for stage, df in reliability.items()]).to_csv(
        out_dir / "reliability_tier_level.csv", index=False, float_format="%.6g")

    summary = {
        "run": run_dir.name, "target": config["target"], "split": SPLIT, "n": int(len(true_tier)),
        "n_true_urgent": int((true_tier == URGENT).sum()),
        "softmax_classes": len(dataset.classes),
        "baseline": escalation_metrics(true_tier, pred_tier, np.zeros(len(true_tier)), 1.0),
        "risk_coverage": risk,
        "calibration": calibration,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, "sweeps": sweeps, "curves": curves, "reliability": reliability,
            "label": f"{config['target']} model", "out_dir": out_dir}


# ---------------------------------------------------------------- figures

def save_figures(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (score, title) in zip(axes, SCORES.items()):
        for colour, res in zip(colours, results):
            sweep = res["sweeps"][score].sort_values("escalation_rate")
            ax.plot(100 * sweep["escalation_rate"], sweep["urgent_sensitivity"], color=colour,
                    label=res["label"], drawstyle="steps-post")
        ax.set(title=f"Escalation by {title}", xlabel="Escalation rate (% of all images)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Urgent sensitivity after escalation")
    axes[0].legend(loc="lower right")
    fig.suptitle("Validation split")
    fig.tight_layout()
    fig.savefig(out_dir / "escalation_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (score, title) in zip(axes, SCORES.items()):
        for colour, res in zip(colours, results):
            curve = next(c for c in res["curves"] if c["score"].iat[0] == score)
            aurc = res["summary"]["risk_coverage"][score]["aurc"]
            ax.plot(curve["coverage"], curve["risk"], color=colour, label=f"{res['label']} (AURC {aurc:.3f})")
        ax.set(title=f"Risk-coverage by {title}", xlabel="Coverage (fraction kept, not escalated)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Tier error rate on kept cases")
    fig.suptitle("Validation split")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 2, figsize=(9, 4.2 * len(results)), squeeze=False)
    for row, res in zip(axes, results):
        cal = res["summary"]["calibration"]
        for ax, stage in zip(row, ("before", "after")):
            bins = res["reliability"][stage]
            filled = bins[bins["count"] > 0]
            centres = (filled["bin_lower"] + filled["bin_upper"]) / 2
            ax.bar(centres, filled["accuracy"], width=1 / ECE_BINS, edgecolor="black", alpha=0.7, label="Accuracy")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ece = cal["tier_level"][f"ece_{stage}"]
            name = "before scaling" if stage == "before" else f"after scaling (T = {cal['temperature']:.2f})"
            ax.set(title=f"{res['label']}, {name}\nTier-level ECE {ece:.3f}", xlim=(0, 1), ylim=(0, 1),
                   xlabel="Confidence (max tier probability)", ylabel="Accuracy")
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Validation split. Temperature fitted on the same split: post-scaling results are optimistic.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)


def print_report(results):
    print(f"Validation split. {SAME_SPLIT_WARNING}\n")
    for res in results:
        s = res["summary"]
        b, cal = s["baseline"], s["calibration"]
        print(f"{res['label']} ({s['run']}), n={s['n']}, true Urgent={s['n_true_urgent']}")
        print(f"  no escalation: Urgent sensitivity {b['urgent_sensitivity']:.3f}, "
              f"Urgent->Routine {b['undertriage_urgent_to_routine']}, missed Urgent {b['missed_urgent_still_missed']}")
        for score, title in SCORES.items():
            r = s["risk_coverage"][score]
            print(f"  {title}: AURC {r['aurc']:.4f}, error on most uncertain 10% {r['error_most_uncertain_10pct']:.3f}, "
                  f"20% {r['error_most_uncertain_20pct']:.3f}")
        print(f"  temperature {cal['temperature']:.3f}; tier-level ECE {cal['tier_level']['ece_before']:.4f} -> "
              f"{cal['tier_level']['ece_after']:.4f}")
        if "condition_level" in cal:
            print(f"  condition-level ECE {cal['condition_level']['ece_before']:.4f} -> "
                  f"{cal['condition_level']['ece_after']:.4f}")
        print(f"  saved to {res['out_dir']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, help="where to save comparison figures "
                        "(default: runs/comparisons/uncertainty_val_<time>)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    out_dir = args.out_dir or REPO_ROOT / "runs" / "comparisons" / f"uncertainty_val_{datetime.now():%Y%m%d-%H%M%S}"
    results = [analyse_run(run_dir, device=args.device) for run_dir in args.run_dirs]
    save_figures(results, out_dir)
    (out_dir / "README.txt").write_text(
        "Runs: " + ", ".join(str(r) for r in args.run_dirs) + f"\nSplit: {SPLIT}\n{SAME_SPLIT_WARNING}\n")
    print_report(results)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
