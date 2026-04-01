"""Launch a supervised fine-tuning run on Tinker.

This is a scaffold, not a final one-click training script. The exact evaluator wiring and
training loop can vary depending on the current cookbook version and your chosen base model.

What this script already does:
- loads config
- validates environment variables
- prints a concrete training plan
- shows the places where Tinker calls should go

Next step for your team:
- copy a minimal SL example from the official Tinker cookbook
- adapt its renderer / data loader to use data/processed/*.jsonl from this repo
- keep the logging and metrics paths from this project
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv



def load_config(path: Path) -> dict:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    config = load_config(root / args.config)
    output_dir = root / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise EnvironmentError("TINKER_API_KEY not found. Copy .env.example to .env and fill it in.")

    plan = {
        "base_model": config["model"]["base_model"],
        "lora_rank": config["model"]["lora_rank"],
        "learning_rate": config["training"]["learning_rate"],
        "num_epochs": config["training"]["num_epochs"],
        "train_file": config["paths"]["train_file"],
        "val_file": config["paths"]["val_file"],
        "output_dir": str(output_dir),
    }

    print("Training plan:")
    print(json.dumps(plan, indent=2))

    notes = [
        "TODO 1: Install `tinker` or `tinker-cookbook` in your environment.",
        "TODO 2: Create a ServiceClient and LoRA TrainingClient.",
        "TODO 3: Build a supervised data loader from data/processed/train.jsonl.",
        "TODO 4: Run cross-entropy supervised updates.",
        "TODO 5: Save checkpoints and collect train/validation losses.",
        "TODO 6: Export a sampling client for test-time predictions.",
    ]
    print("\n".join(notes))

    # -------------------------------------------------------------------------
    # Minimal reference shape based on the official Tinker docs:
    #
    # import tinker
    # service_client = tinker.ServiceClient()
    # training_client = service_client.create_lora_training_client(
    #     base_model=config["model"]["base_model"],
    #     rank=config["model"]["lora_rank"],
    # )
    #
    # Then adapt a supervised loop from the official cookbook so each datum is rendered
    # from one JSONL record in data/processed/train.jsonl.
    # -------------------------------------------------------------------------

    summary_path = output_dir / "run_plan.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved run plan to {summary_path}")


if __name__ == "__main__":
    main()
