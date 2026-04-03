"""Plot training curves from Tinker metrics.jsonl and compare baseline/val/test metrics.

Usage:
    python scripts/07_plot_curves.py --run-dir results/runs/lora_base_experiment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    metrics_path = run_dir / "metrics.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {metrics_path}")

    rows = load_jsonl(metrics_path)
    df = pd.DataFrame(rows)

    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Plot 1: training NLL over steps ----------
    if "train_mean_nll" in df.columns:
        train_df = df[df["train_mean_nll"].notna()].copy()
        train_df["step"] = range(1, len(train_df) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_df["step"], train_df["train_mean_nll"])
        plt.xlabel("Step")
        plt.ylabel("train_mean_nll")
        plt.title("Training NLL Curve")
        plt.tight_layout()
        plt.savefig(out_dir / "train_nll_curve.png", dpi=200)
        plt.close()

    # ---------- Plot 2: eval NLL points if available ----------
    if "test/nll" in df.columns:
        eval_df = df[df["test/nll"].notna()].copy()
        if not eval_df.empty:
            eval_df["eval_index"] = range(1, len(eval_df) + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(eval_df["eval_index"], eval_df["test/nll"], marker="o")
            plt.xlabel("Evaluation Event")
            plt.ylabel("test/nll")
            plt.title("Evaluation NLL During Training")
            plt.tight_layout()
            plt.savefig(out_dir / "eval_nll_curve.png", dpi=200)
            plt.close()

    # ---------- Plot 3: summary bar chart ----------
    summary_rows = []

    baseline_path = root / "results" / "tables" / "val_predictions_200_metrics.json"
    val_path = root / "results" / "tables" / "val_predictions_full_metrics.json"
    test_path = root / "results" / "tables" / "test_predictions_full_metrics.json"

    for name, path in [
        ("baseline_subset", baseline_path),
        ("validation_full", val_path),
        ("test_full", test_path),
    ]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            summary_rows.append(
                {
                    "split": name,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        plt.figure(figsize=(8, 5))
        x = range(len(summary_df))
        width = 0.35

        plt.bar([i - width / 2 for i in x], summary_df["accuracy"], width=width, label="accuracy")
        plt.bar([i + width / 2 for i in x], summary_df["macro_f1"], width=width, label="macro_f1")

        plt.xticks(list(x), summary_df["split"], rotation=15)
        plt.ylabel("Score")
        plt.title("Model Performance Summary")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "performance_summary.png", dpi=200)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()