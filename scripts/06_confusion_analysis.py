"""Create a confusion matrix and highlight the most confused intent pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.metrics import build_confusion_df



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_path = root / args.predictions

    with pred_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows)
    labels = sorted(set(df["gold_label"]).union(set(df["predicted_label"])))
    confusion = build_confusion_df(df["gold_label"], df["predicted_label"], labels)

    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    confusion.to_csv(out_dir / "confusion_matrix.csv")

    pairs = []
    for gold in labels:
        for pred in labels:
            if gold == pred:
                continue
            count = int(confusion.loc[gold, pred])
            if count > 0:
                pairs.append({"gold": gold, "pred": pred, "count": count})

    top_pairs = pd.DataFrame(pairs).sort_values("count", ascending=False).head(args.top_k)
    top_pairs.to_csv(out_dir / "top_confused_pairs.csv", index=False)

    print(top_pairs)
    print(f"Saved confusion outputs to {out_dir}")


if __name__ == "__main__":
    main()
