"""Dataset utilities for CLINC150 preparation and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def ensure_parent(path: Path) -> None:
    """Create the parent directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)



def write_jsonl(records: Iterable[dict], path: Path) -> None:
    """Write an iterable of dictionaries to JSONL."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into memory."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]



def normalize_text(text: str) -> str:
    """Normalize whitespace in a user utterance.

    This keeps preprocessing intentionally light so we do not destroy signal in short
    conversational requests.
    """
    return " ".join(str(text).strip().split())



def save_label_metadata(labels: List[str], output_dir: Path) -> None:
    """Save label list and label-to-index mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = {label: idx for idx, label in enumerate(sorted(labels))}

    with (output_dir / "intent_list.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(labels), f, indent=2, ensure_ascii=False)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)



def load_processed_split(path: Path) -> pd.DataFrame:
    """Load processed JSONL into a dataframe useful for analysis."""
    rows = read_jsonl(path)
    return pd.DataFrame(
        {
            "utterance": [row["meta"]["utterance"] for row in rows],
            "label": [row["meta"]["label"] for row in rows],
            "split": [row["meta"]["split"] for row in rows],
        }
    )
