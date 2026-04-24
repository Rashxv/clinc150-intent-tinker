"""Run supervised fine-tuning on the CLINC150 processed JSONL using Tinker Cookbook.

Usage:
    python scripts/04_train_tinker.py --config configs/base.yaml
    python scripts/04_train_tinker.py --config configs/final_run.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import chz
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def load_config(path: Path) -> dict:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_config(config: dict, root: Path) -> train.Config:
    """Build a Tinker Cookbook supervised training config from repo YAML."""
    model_name = config["model"]["base_model"]
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    train_file = config["paths"]["train_file"].replace("\\", "/")
    output_dir = str((root / config["paths"]["output_dir"]).resolve())

    batch_size = int(config["training"]["batch_size"]) * int(config["training"]["grad_accum_steps"])

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=train_file,
    )

    overrides = {
        "log_path": output_dir,
        "model_name": model_name,
        "renderer_name": renderer_name,
        "dataset_builder": dataset_builder,
        "learning_rate": float(config["training"]["learning_rate"]),
        "lr_schedule": "linear",
        "num_epochs": int(config["training"]["num_epochs"]),
        "lora_rank": int(config["model"]["lora_rank"]),
        "eval_every": int(config["training"]["eval_every_steps"]),
        "save_every": int(config["training"]["save_every_steps"]),
    }

    return chz.Blueprint(train.Config).apply(overrides).make()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    if not os.getenv("TINKER_API_KEY"):
        raise EnvironmentError("TINKER_API_KEY not found in .env or environment.")

    repo_config = load_config(root / args.config)
    output_dir = root / repo_config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "base_model": repo_config["model"]["base_model"],
        "lora_rank": repo_config["model"]["lora_rank"],
        "learning_rate": repo_config["training"]["learning_rate"],
        "num_epochs": repo_config["training"]["num_epochs"],
        "train_file": repo_config["paths"]["train_file"],
        "val_file": repo_config["paths"]["val_file"],
        "output_dir": str(output_dir),
    }

    print("Training plan:")
    print(json.dumps(plan, indent=2))

    with (output_dir / "run_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    train_config = build_train_config(repo_config, root)

    # Ask before deleting/resuming existing log dir
    cli_utils.check_log_dir(train_config.log_path, behavior_if_exists="ask")

    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    main()