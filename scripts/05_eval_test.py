"""Evaluate saved predictions against the processed test split.

Expected prediction format (JSONL):
    {"utterance": "...", "gold_label": "...", "predicted_label": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.metrics import build_per_intent_report, compute_summary_metrics, save_metrics_json



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_path = root / args.predictions

    with pred_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows)
    required = {"gold_label", "predicted_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predictions file must contain columns: {sorted(required)}")

    metrics = compute_summary_metrics(df["gold_label"], df["predicted_label"])
    print(json.dumps(metrics, indent=2))

    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = pred_path.stem
    save_metrics_json(metrics, out_dir / f"{stem}_metrics.json")
    build_per_intent_report(df["gold_label"], df["predicted_label"]).to_csv(
        out_dir / f"{stem}_per_intent_report.csv", index=False
    )

    print(f"Saved metrics and per-intent report to {out_dir}")


if __name__ == "__main__":
    main()
