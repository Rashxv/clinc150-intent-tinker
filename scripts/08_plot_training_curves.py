"""Plot training loss curves for all fine-tuning experiments and final accuracy.

Usage:
    python scripts/08_plot_training_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

EXPERIMENTS = [
    {
        "label": "Exp A: rank=32, LR=5e-4, ep=2",
        "run_dir": "results/runs/lora_base_experiment",
        "metrics_json": "results/tables/test_predictions_full_metrics.json",
    },
    {
        "label": "Exp B: rank=8,  LR=5e-4, ep=2",
        "run_dir": "results/runs/exp_b_rank8",
        "metrics_json": "results/tables/exp_b_test_predictions_metrics.json",
    },
    {
        "label": "Exp C: rank=64, LR=5e-4, ep=2",
        "run_dir": "results/runs/exp_c_rank64",
        "metrics_json": "results/tables/exp_c_test_predictions_metrics.json",
    },
    {
        "label": "Exp D: rank=32, LR=1e-4, ep=3",
        "run_dir": "results/runs/exp_d_lr1e4_e3",
        "metrics_json": "results/tables/exp_d_test_predictions_metrics.json",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, :])   # full top row: loss curves
    ax_acc  = fig.add_subplot(gs[1, 0])   # bottom left: accuracy bar
    ax_f1   = fig.add_subplot(gs[1, 1])   # bottom right: macro-F1 bar

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    acc_labels, acc_vals, f1_vals = [], [], []

    for exp, color in zip(EXPERIMENTS, colors):
        run_path = ROOT / exp["run_dir"] / "metrics.jsonl"
        metrics_path = ROOT / exp["metrics_json"]

        # --- Loss curve ---
        if run_path.exists():
            rows = load_jsonl(run_path)
            df = pd.DataFrame(rows)
            if "train_mean_nll" in df.columns and "progress" in df.columns:
                train_df = df[df["train_mean_nll"].notna()].copy()
                # smooth with a rolling window so the curve is readable
                smoothed = train_df["train_mean_nll"].rolling(window=5, min_periods=1).mean()
                ax_loss.plot(
                    train_df["progress"] * 100,
                    smoothed,
                    label=exp["label"],
                    color=color,
                    linewidth=1.8,
                )
        else:
            print(f"Warning: missing run dir {run_path}")

        # --- Final metrics ---
        if metrics_path.exists():
            m = load_json(metrics_path)
            acc = m.get("accuracy", m.get("test_accuracy", 0.0))
            f1  = m.get("macro_f1", m.get("test_macro_f1", 0.0))
            acc_labels.append(exp["label"].split(":")[0])  # "Exp A" etc.
            acc_vals.append(acc)
            f1_vals.append(f1)
        else:
            print(f"Warning: missing metrics file {metrics_path}")

    # --- Dress up loss plot ---
    ax_loss.set_xlabel("Training Progress (%)", fontsize=11)
    ax_loss.set_ylabel("Train NLL (loss)", fontsize=11)
    ax_loss.set_title("Training Loss over Time — All Experiments", fontsize=13)
    ax_loss.legend(fontsize=9)
    ax_loss.set_xlim(0, 100)
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(axis="y", alpha=0.3)

    # --- Accuracy bar ---
    bars = ax_acc.bar(acc_labels, acc_vals, color=colors[: len(acc_labels)])
    ax_acc.set_ylabel("Test Accuracy", fontsize=11)
    ax_acc.set_title("Final Test Accuracy", fontsize=12)
    ax_acc.set_ylim(0.93, 0.98)
    ax_acc.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, acc_vals):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # --- Macro-F1 bar ---
    bars = ax_f1.bar(acc_labels, f1_vals, color=colors[: len(acc_labels)])
    ax_f1.set_ylabel("Test Macro-F1", fontsize=11)
    ax_f1.set_title("Final Test Macro-F1", fontsize=12)
    ax_f1.set_ylim(0.93, 0.98)
    ax_f1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, f1_vals):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    out_path = out_dir / "training_curves_all_experiments.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
