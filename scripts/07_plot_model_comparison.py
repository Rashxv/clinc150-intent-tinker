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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics: dict, primary: str, fallback: str) -> float:
    value = metrics.get(primary, metrics.get(fallback))
    if value is None:
        raise KeyError(f"Could not find '{primary}' or '{fallback}' in metrics file.")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert-metrics",
        type=str,
        default="results/bert_baseline/metrics.json",
    )
    parser.add_argument(
        "--roberta-metrics",
        type=str,
        default="results/roberta_baseline/metrics.json",
    )
    parser.add_argument(
        "--llama-ft-metrics",
        type=str,
        default="results/tables/llama_ft_test_predictions_metrics.json",
    )
    parser.add_argument(
        "--llama-zero-shot-metrics",
        type=str,
        default="results/llama_3_2_1b_zero_shot_predictions_metrics.json",
    )
    parser.add_argument(
        "--llama-few-shot-metrics",
        type=str,
        default="results/llama_3_2_1b_few_shot_predictions_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    bert_path = root / args.bert_metrics
    roberta_path = root / args.roberta_metrics
    llama_ft_path = root / args.llama_ft_metrics
    llama_zero_path = root / args.llama_zero_shot_metrics
    llama_few_path = root / args.llama_few_shot_metrics
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "BERT": bert_path,
        "RoBERTa": roberta_path,
        "Fine-tuned Llama": llama_ft_path,
        "Llama zero-shot": llama_zero_path,
        "Llama few-shot": llama_few_path,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} metrics file not found: {path}")

    bert = load_json(bert_path)
    roberta = load_json(roberta_path)
    llama_ft = load_json(llama_ft_path)
    llama_zero = load_json(llama_zero_path)
    llama_few = load_json(llama_few_path)

    rows = [
        {
            "model": "BERT",
            "accuracy": get_metric(bert, "test_accuracy", "accuracy"),
            "macro_f1": get_metric(bert, "test_macro_f1", "macro_f1"),
        },
        {
            "model": "RoBERTa",
            "accuracy": get_metric(roberta, "test_accuracy", "accuracy"),
            "macro_f1": get_metric(roberta, "test_macro_f1", "macro_f1"),
        },
        {
            "model": "Llama zero-shot",
            "accuracy": get_metric(llama_zero, "accuracy", "test_accuracy"),
            "macro_f1": get_metric(llama_zero, "macro_f1", "test_macro_f1"),
        },
        {
            "model": "Llama few-shot",
            "accuracy": get_metric(llama_few, "accuracy", "test_accuracy"),
            "macro_f1": get_metric(llama_few, "macro_f1", "test_macro_f1"),
        },
        {
            "model": "Fine-tuned Llama",
            "accuracy": get_metric(llama_ft, "accuracy", "test_accuracy"),
            "macro_f1": get_metric(llama_ft, "macro_f1", "test_macro_f1"),
        },
    ]

    df = pd.DataFrame(rows)

    summary_path = out_dir / "model_comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], df["accuracy"], width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], df["macro_f1"], width=width, label="Macro-F1")
    plt.xticks(list(x), df["model"], rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("BERT vs RoBERTa vs Llama Baselines vs Fine-tuned Llama")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["model"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["model"], df["macro_f1"], marker="o", label="Macro-F1")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_line.png", dpi=200)
    plt.close()

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {out_dir / 'model_comparison_bar.png'}")
    print(f"- {out_dir / 'model_comparison_line.png'}")


if __name__ == "__main__":
    main()
