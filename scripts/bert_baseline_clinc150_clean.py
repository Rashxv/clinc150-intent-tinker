from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging as hf_logging

# Quiet down noisy Transformers / HF logs
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl


def get_utterance(row: dict) -> str:
    meta = row.get("meta", {})
    if "utterance" in meta:
        return meta["utterance"]

    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        utterance = get_utterance(row).strip()
        label = row.get("meta", {}).get("label", "").strip()
        if utterance and label:
            records.append({"utterance": utterance, "label": label})
    return pd.DataFrame(records)


class IntentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.utterances = dataframe["utterance"].tolist()
        self.labels = [label2id[label] for label in dataframe["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        encoded = self.tokenizer(
            self.utterances[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_training_arguments(
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    weight_decay: float,
    seed: int,
    use_fp16: bool,
) -> TrainingArguments:
    params = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "weight_decay": weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "seed": seed,
    }

    if "overwrite_output_dir" in params:
        kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "no"

    if "save_total_limit" in params:
        kwargs["save_total_limit"] = 2

    if "report_to" in params:
        kwargs["report_to"] = []

    if "fp16" in params:
        kwargs["fp16"] = use_fp16

    if "dataloader_num_workers" in params:
        kwargs["dataloader_num_workers"] = 0

    if "disable_tqdm" in params:
        kwargs["disable_tqdm"] = False

    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="Hugging Face model name. Use roberta-base if you want RoBERTa.",
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/train.jsonl",
        help="Training split file",
    )
    parser.add_argument(
        "--val-file",
        default="data/processed/val.jsonl",
        help="Validation split file. If missing, a split will be made from training data.",
    )
    parser.add_argument(
        "--test-file",
        default="data/processed/test.jsonl",
        help="Test split file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/bert_baseline",
        help="Directory for model checkpoints and outputs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum token length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {ROOT}")
    print(f"Model: {args.model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Loading data...")

    train_df = rows_to_dataframe(read_jsonl(ROOT / args.train_file))

    val_path = ROOT / args.val_file
    if val_path.exists():
        val_df = rows_to_dataframe(read_jsonl(val_path))
        print(f"Loaded validation file: {val_path}")
    else:
        print("Validation file not found. Creating stratified validation split from train...")
        train_df, val_df = train_test_split(
            train_df,
            test_size=0.1,
            stratify=train_df["label"],
            random_state=args.seed,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    test_df = rows_to_dataframe(read_jsonl(ROOT / args.test_file))

    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")

    all_labels = sorted(set(train_df["label"]) | set(val_df["label"]) | set(test_df["label"]))
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"Number of labels: {len(all_labels)}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = IntentDataset(train_df, tokenizer, label2id, args.max_length)
    val_dataset = IntentDataset(val_df, tokenizer, label2id, args.max_length)
    test_dataset = IntentDataset(test_df, tokenizer, label2id, args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    use_fp16 = torch.cuda.is_available()

    training_args = make_training_arguments(
        output_dir=output_dir / "checkpoints",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_fp16=use_fp16,
    )

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "callbacks" in trainer_params:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(**trainer_kwargs)

    print("Training...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)

    print("Predicting on test set...")
    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    test_label_ids = test_output.label_ids
    test_pred_ids = np.argmax(test_logits, axis=-1)

    gold_labels = [id2label[int(idx)] for idx in test_label_ids]
    predicted_labels = [id2label[int(idx)] for idx in test_pred_ids]

    test_accuracy = accuracy_score(gold_labels, predicted_labels)
    test_macro_f1 = f1_score(gold_labels, predicted_labels, average="macro")

    metrics = {
        "model_name": args.model_name,
        "num_labels": len(all_labels),
        "train_examples": len(train_df),
        "validation_examples": len(val_df),
        "test_examples": len(test_df),
        "validation_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
        "validation_macro_f1": float(val_metrics.get("eval_macro_f1", 0.0)),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
    }

    predictions_path = output_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for utterance, gold, pred in zip(test_df["utterance"], gold_labels, predicted_labels):
            record = {
                "utterance": utterance,
                "gold_label": gold,
                "predicted_label": pred,
                "model_name": args.model_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = output_dir / "per_intent_report.csv"
    report_df.to_csv(report_csv_path, index=True)

    metrics_path = output_dir / "metrics.json"
    save_json(metrics, metrics_path)

    model_dir = output_dir / "best_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print("\nFinished.")
    print(f"Validation Accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"Validation Macro-F1: {metrics['validation_macro_f1']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Macro-F1: {metrics['test_macro_f1']:.4f}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Per-intent report saved to: {report_csv_path}")
    print(f"Best model saved to: {model_dir}")


if __name__ == "__main__":
    main()
