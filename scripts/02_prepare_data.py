"""Prepare CLINC150 as instruction-response JSONL files.

Usage:
    python scripts/02_prepare_data.py --include-oos false
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.dataset_utils import normalize_text, save_label_metadata, write_jsonl
from src.prompts import build_chat_example


def parse_bool(text: str) -> bool:
    """Parse a human-friendly boolean string."""
    return text.strip().lower() in {"1", "true", "yes", "y"}



def to_records(split_ds, split_name: str, include_oos: bool, labels: list[str]) -> list[dict]:
    """Convert one HF split into chat-style records."""
    records = []
    for row in split_ds:
        utterance = normalize_text(row["text"])
        label = row["intent"]

        if (not include_oos) and label == "oos":
            continue

        records.append(
            build_chat_example(
                utterance=utterance,
                label=label,
                labels=labels,
                split=split_name,
                include_label_list=True,
            )
        )
    return records



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-oos", type=str, default="false")
    args = parser.parse_args()

    include_oos = parse_bool(args.include_oos)

    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    metadata_dir = root / "data" / "metadata"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("DeepPavlov/clinc150")

    raw_labels = sorted(set(dataset["train"]["intent"]))
    labels = raw_labels if include_oos else [x for x in raw_labels if x != "oos"]
    save_label_metadata(labels, metadata_dir)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for hf_split, short_name in split_map.items():
        records = to_records(dataset[hf_split], short_name, include_oos, labels)
        out_path = processed_dir / f"{short_name}.jsonl"
        write_jsonl(records, out_path)
        print(f"Wrote {len(records)} examples -> {out_path}")

    print(f"Done. include_oos={include_oos}")


if __name__ == "__main__":
    main()
