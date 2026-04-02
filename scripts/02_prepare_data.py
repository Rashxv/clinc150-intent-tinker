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


def detect_columns(split_ds) -> tuple[str, str]:
    """Detect text and label columns from a HF dataset split."""
    column_names = split_ds.column_names

    # Text column
    for candidate in ("text", "utterance", "sentence", "query"):
        if candidate in column_names:
            text_col = candidate
            break
    else:
        raise ValueError(f"Could not find a text column. Found columns: {column_names}")

    # Label column
    for candidate in ("intent", "label", "labels", "category"):
        if candidate in column_names:
            label_col = candidate
            break
    else:
        raise ValueError(f"Could not find a label column. Found columns: {column_names}")

    return text_col, label_col


def build_label_decoder(split_ds, label_col: str):
    """Return label_names and a decoder function for raw label values."""
    label_feature = split_ds.features[label_col]

    # Case 1: Hugging Face ClassLabel
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = list(label_feature.names)

        def decode_label(value):
            if isinstance(value, int):
                return label_names[value]
            return str(value)

        return label_names, decode_label

    # Case 2: already strings or plain values
    raw_values = split_ds[label_col]
    label_names = sorted({str(x) for x in raw_values})

    def decode_label(value):
        return str(value)

    return label_names, decode_label


def to_records(
    split_ds,
    split_name: str,
    include_oos: bool,
    labels: list[str],
    text_col: str,
    label_col: str,
    decode_label,
) -> list[dict]:
    """Convert one HF split into chat-style records."""
    records = []

    for row in split_ds:
        utterance = normalize_text(str(row[text_col]))
        label = decode_label(row[label_col])

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

    train_split = dataset["train"]
    text_col, label_col = detect_columns(train_split)
    label_names, decode_label = build_label_decoder(train_split, label_col)

    print(f"Detected text column: {text_col}")
    print(f"Detected label column: {label_col}")
    print(f"Number of labels found: {len(label_names)}")

    labels = label_names if include_oos else [x for x in label_names if x != "oos"]
    save_label_metadata(labels, metadata_dir)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for hf_split, short_name in split_map.items():
        records = to_records(
            split_ds=dataset[hf_split],
            split_name=short_name,
            include_oos=include_oos,
            labels=labels,
            text_col=text_col,
            label_col=label_col,
            decode_label=decode_label,
        )
        out_path = processed_dir / f"{short_name}.jsonl"
        write_jsonl(records, out_path)
        print(f"Wrote {len(records)} examples -> {out_path}")

    print(f"Done. include_oos={include_oos}")


if __name__ == "__main__":
    main()