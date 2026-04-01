"""Download the CLINC150 dataset from Hugging Face and save CSV snapshots.

Usage:
    python scripts/01_load_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("DeepPavlov/clinc150")

    for split_name, split_ds in dataset.items():
        out_path = raw_dir / f"clinc150_{split_name}.csv"
        split_ds.to_csv(str(out_path), index=False)
        print(f"Saved {split_name} -> {out_path}")

    print("Finished downloading CLINC150.")


if __name__ == "__main__":
    main()
