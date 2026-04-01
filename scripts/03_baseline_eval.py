"""A simple local baseline script.

This script is intentionally lightweight. It does NOT call a hosted base model yet.
Instead, it provides:
1. a majority-class baseline
2. a random-label baseline

You can later extend it to query a base model through Tinker sampling or another API.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl
from src.metrics import build_per_intent_report, compute_summary_metrics



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    train_rows = read_jsonl(root / "data" / "processed" / "train.jsonl")
    eval_rows = read_jsonl(root / "data" / "processed" / f"{args.split}.jsonl")

    train_labels = [row["meta"]["label"] for row in train_rows]
    eval_labels = [row["meta"]["label"] for row in eval_rows]
    allowed = sorted(set(train_labels))

    majority_label = Counter(train_labels).most_common(1)[0][0]

    y_pred_majority = [majority_label for _ in eval_labels]
    y_pred_random = [random.choice(allowed) for _ in eval_labels]

    results = {
        "majority": compute_summary_metrics(eval_labels, y_pred_majority),
        "random": compute_summary_metrics(eval_labels, y_pred_random),
    }

    print(json.dumps(results, indent=2))

    report = build_per_intent_report(eval_labels, y_pred_majority)
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / f"baseline_majority_per_intent_{args.split}.csv", index=False)
    print(f"Saved per-intent majority report to {out_dir}")


if __name__ == "__main__":
    main()
