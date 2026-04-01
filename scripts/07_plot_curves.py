"""Plot train/validation curves from a CSV log file.

Expected CSV columns:
- step
- train_loss
- val_loss
- optional: val_accuracy
- optional: val_macro_f1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-csv", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_path = root / args.log_csv
    df = pd.read_csv(log_path)

    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["train_loss"], label="train_loss")
    plt.plot(df["step"], df["val_loss"], label="val_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()

    for metric in ["val_accuracy", "val_macro_f1"]:
        if metric in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(df["step"], df[metric])
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.title(metric)
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}.png", dpi=200)
            plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
