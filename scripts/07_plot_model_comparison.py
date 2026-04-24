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
        default=None,
        help="Optional path to RoBERTa metrics JSON; omit to skip.",
    )
    parser.add_argument(
        "--llama-zero-shot-metrics",
        type=str,
        default="results/llama_3_2_1b_instruct_zero_shot_predictions_metrics.json",
    )
    parser.add_argument(
        "--llama-few-shot-metrics",
        type=str,
        default="results/llama_3_2_1b_instruct_few_shot_predictions_metrics.json",
    )
    # Fine-tuned experiment metrics — add more with --ft-extra
    parser.add_argument(
        "--ft-a-metrics",
        type=str,
        default="results/tables/test_predictions_full_metrics.json",
        help="Exp A: rank=32, LR=5e-4, epochs=2 (original run)",
    )
    parser.add_argument(
        "--ft-b-metrics",
        type=str,
        default=None,
        help="Exp B: rank=8, LR=5e-4, epochs=2",
    )
    parser.add_argument(
        "--ft-c-metrics",
        type=str,
        default=None,
        help="Exp C: rank=64, LR=5e-4, epochs=2",
    )
    parser.add_argument(
        "--ft-d-metrics",
        type=str,
        default=None,
        help="Exp D: rank=32, LR=1e-4, epochs=3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Required files ---
    bert_path = root / args.bert_metrics
    llama_zero_path = root / args.llama_zero_shot_metrics
    llama_few_path = root / args.llama_few_shot_metrics

    required = {
        "BERT": bert_path,
        "Llama zero-shot": llama_zero_path,
        "Llama few-shot": llama_few_path,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} metrics file not found: {path}")

    # --- Optional files ---
    roberta_path = root / args.roberta_metrics if args.roberta_metrics else None

    ft_experiments = [
        ("Llama FT (rank=32, LR=5e-4, ep=2)", args.ft_a_metrics),
        ("Llama FT (rank=8,  LR=5e-4, ep=2)", args.ft_b_metrics),
        ("Llama FT (rank=64, LR=5e-4, ep=2)", args.ft_c_metrics),
        ("Llama FT (rank=32, LR=1e-4, ep=3)", args.ft_d_metrics),
    ]

    # --- Build rows ---
    bert = load_json(bert_path)
    llama_zero = load_json(llama_zero_path)
    llama_few = load_json(llama_few_path)

    rows = [
        {
            "model": "BERT",
            "accuracy": get_metric(bert, "test_accuracy", "accuracy"),
            "macro_f1": get_metric(bert, "test_macro_f1", "macro_f1"),
        },
    ]

    if roberta_path is not None and roberta_path.exists():
        roberta = load_json(roberta_path)
        rows.append({
            "model": "RoBERTa",
            "accuracy": get_metric(roberta, "test_accuracy", "accuracy"),
            "macro_f1": get_metric(roberta, "test_macro_f1", "macro_f1"),
        })

    rows += [
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
    ]

    for label, path_str in ft_experiments:
        if path_str is None:
            continue
        path = root / path_str
        if not path.exists():
            print(f"Skipping {label}: file not found ({path})")
            continue
        m = load_json(path)
        rows.append({
            "model": label,
            "accuracy": get_metric(m, "accuracy", "test_accuracy"),
            "macro_f1": get_metric(m, "macro_f1", "test_macro_f1"),
        })

    df = pd.DataFrame(rows)

    summary_path = out_dir / "model_comparison_summary.csv"
    df.to_csv(summary_path, index=False)
    print(df.to_string(index=False))

    x = range(len(df))
    width = 0.35
    fig_w = max(10, len(df) * 1.6)

    plt.figure(figsize=(fig_w, 5))
    plt.bar([i - width / 2 for i in x], df["accuracy"], width=width, label="Accuracy")
    plt.bar([i + width / 2 for i in x], df["macro_f1"], width=width, label="Macro-F1")
    plt.xticks(list(x), df["model"], rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Model Comparison — CLINC150 Intent Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 5))
    plt.plot(df["model"], df["accuracy"], marker="o", label="Accuracy")
    plt.plot(df["model"], df["macro_f1"], marker="o", label="Macro-F1")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison — CLINC150")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison_line.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print(f"- {summary_path}")
    print(f"- {out_dir / 'model_comparison_bar.png'}")
    print(f"- {out_dir / 'model_comparison_line.png'}")


if __name__ == "__main__":
    main()
