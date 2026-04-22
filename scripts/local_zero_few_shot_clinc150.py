from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl
from src.metrics import build_per_intent_report, compute_summary_metrics, save_metrics_json


def get_utterance(row: dict) -> str:
    meta = row.get("meta", {})
    if "utterance" in meta:
        return meta["utterance"]

    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r'Utterance:\s*"(.*?)"', content, flags=re.DOTALL)
            return match.group(1) if match else content

    return ""


def build_label_set(*rows_groups: Iterable[dict]) -> list[str]:
    labels: set[str] = set()
    for rows in rows_groups:
        for row in rows:
            label = row.get("meta", {}).get("label", "").strip()
            if label:
                labels.add(label)
    return sorted(labels)


def group_examples_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("meta", {}).get("label", "").strip()
        if label:
            grouped[label].append(row)
    return grouped


def choose_few_shot_examples(train_rows: list[dict], max_examples: int = 4) -> list[dict]:
    grouped = group_examples_by_label(train_rows)
    labels = sorted(grouped.keys())

    support = []
    label_idx = 0

    while len(support) < max_examples and labels:
        label = labels[label_idx % len(labels)]
        examples = grouped[label]
        take_idx = label_idx // len(labels)

        if take_idx < len(examples):
            support.append(examples[take_idx])

        label_idx += 1

        if label_idx > len(labels) * max(len(v) for v in grouped.values()):
            break

    return support[:max_examples]


def build_zero_shot_messages(utterance: str, labels: list[str]) -> list[dict]:
    label_text = ", ".join(labels)
    return [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. "
                "Choose exactly one label from the allowed labels. "
                "Reply with exactly one label from the allowed labels. "
                "Do not explain. Do not add any other words.\n"
                f"Allowed labels: {label_text}"
            ),
        },
        {
            "role": "user",
            "content": f'Utterance: "{utterance}"',
        },
    ]


def build_few_shot_messages(
    utterance: str,
    labels: list[str],
    support_examples: list[dict],
) -> list[dict]:
    label_text = ", ".join(labels)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. "
                "Choose exactly one label from the allowed labels. "
                "Reply with exactly one label from the allowed labels. "
                "Do not explain. Do not add any other words.\n"
                f"Allowed labels: {label_text}"
            ),
        }
    ]

    for ex in support_examples:
        ex_utt = get_utterance(ex)
        ex_label = ex.get("meta", {}).get("label", "").strip()

        messages.append({"role": "user", "content": f'Utterance: "{ex_utt}"'})
        messages.append({"role": "assistant", "content": ex_label})

    messages.append({"role": "user", "content": f'Utterance: "{utterance}"'})

    return messages


def render_prompt(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_parts = []
    for msg in messages:
        prompt_parts.append(f"{msg['role'].upper()}: {msg['content']}")
    prompt_parts.append("ASSISTANT:")
    return "\n\n".join(prompt_parts)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_label(predicted_text: str, all_labels: list[str]) -> str:
    text = predicted_text.strip()
    normalized = normalize_text(text)

    for label in all_labels:
        if normalized == label.lower():
            return label

    cleaned = text.strip().strip('"').strip("'").strip()
    cleaned_norm = normalize_text(cleaned)
    for label in all_labels:
        if cleaned_norm == label.lower():
            return label

    return cleaned if cleaned else text


def generate_label(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B",
        help="Local causal LM for zero-shot/few-shot",
    )
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot"],
        required=True,
        help="Choose zero-shot or few-shot prompting",
    )
    parser.add_argument(
        "--input-file",
        default="data/processed/test.jsonl",
        help="Evaluation split file",
    )
    parser.add_argument(
        "--train-file",
        default="data/processed/train.jsonl",
        help="Training split used only for few-shot support examples",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to save predictions JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N examples for debugging",
    )
    parser.add_argument(
        "--max-few-shot-examples",
        type=int,
        default=4,
        help="Maximum number of few-shot support examples",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
        help="Maximum generated tokens for the predicted label",
    )
    args = parser.parse_args()

    if args.output_file is None:
        safe_mode = args.mode.replace("-", "_")
        safe_model = args.model_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()
        args.output_file = f"results/{safe_model}_{safe_mode}_predictions.jsonl"

    print(f"Project root: {ROOT}")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading evaluation data...")
    eval_rows_all = read_jsonl(ROOT / args.input_file)
    train_rows = read_jsonl(ROOT / args.train_file) if args.mode == "few-shot" else []

    labels = build_label_set(eval_rows_all, train_rows)
    eval_rows = eval_rows_all[: args.limit] if args.limit is not None else eval_rows_all

    print(f"Loaded {len(eval_rows)} evaluation example(s)")
    print(f"Loaded {len(labels)} label(s)")

    support_examples = []
    if args.mode == "few-shot":
        support_examples = choose_few_shot_examples(
            train_rows=train_rows,
            max_examples=args.max_few_shot_examples,
        )
        print(f"Using {len(support_examples)} few-shot support example(s)")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model_kwargs = {}
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model_kwargs["dtype"] = torch.bfloat16
        else:
            model_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    out_path = ROOT / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_labels = []
    predicted_labels = []

    start_time = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(eval_rows, desc=f"Running {args.mode}", unit="example"):
            utterance = get_utterance(row)
            gold_label = row.get("meta", {}).get("label", "").strip()

            if args.mode == "zero-shot":
                messages = build_zero_shot_messages(utterance, labels)
            else:
                messages = build_few_shot_messages(utterance, labels, support_examples)

            prompt_text = render_prompt(tokenizer, messages)

            raw_prediction = generate_label(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            predicted_label = extract_label(raw_prediction, labels)

            record = {
                "utterance": utterance,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "raw_prediction": raw_prediction,
                "mode": args.mode,
                "model_name": args.model_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            gold_labels.append(gold_label)
            predicted_labels.append(predicted_label)

    metrics = compute_summary_metrics(gold_labels, predicted_labels)
    report_df = build_per_intent_report(gold_labels, predicted_labels)

    metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
    report_path = out_path.with_name(out_path.stem + "_per_intent.csv")

    save_metrics_json(metrics, metrics_path)
    report_df.to_csv(report_path, index=False)

    elapsed = time.time() - start_time

    print("\nFinished.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Predictions saved to: {out_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Per-intent report saved to: {report_path}")
    print(f"Total time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
