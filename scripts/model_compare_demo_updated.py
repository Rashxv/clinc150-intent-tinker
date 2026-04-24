from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import tinker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl

load_dotenv(ROOT / ".env")

# Fine-tuned Tinker model
MODEL_PATH = "tinker://6cc3a560-3451-582c-8baf-dca20d7a7dff:train:0/sampler_weights/final"

# Base Llama used for zero-shot / few-shot
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Local BERT classifier path
BERT_MODEL_PATH = ROOT / "results" / "bert_baseline" / "best_model"


def load_rows():
    rows = []
    for path in [
        ROOT / "data/processed/train.jsonl",
        ROOT / "data/processed/val.jsonl",
        ROOT / "data/processed/test.jsonl",
    ]:
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows


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


def build_label_info():
    rows = load_rows()
    labels = []
    for row in rows:
        label = row.get("meta", {}).get("label", "").strip()
        if label:
            labels.append(label)

    unique_labels = sorted(set(labels))
    return unique_labels


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_label(predicted_text, all_labels):
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

    chunks = re.split(r"[\n,;:]+", text)
    for chunk in chunks:
        candidate = chunk.strip().strip('"').strip("'").strip()
        candidate_norm = normalize_text(candidate)
        for label in all_labels:
            if candidate_norm == label.lower():
                return label

    for label in sorted(all_labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label.lower())}\b", normalized):
            return label

    first_line = text.splitlines()[0].strip() if text else ""
    return first_line


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


def generate_local_label(model, tokenizer, prompt_text: str, max_new_tokens: int, device: torch.device) -> str:
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


def build_tinker_prompt(user_text, labels):
    label_text = ", ".join(labels)
    return (
        "System: You are an intent classifier. "
        "Choose exactly one label from the allowed labels. "
        "Output only the label and nothing else.\n\n"
        f'User: Utterance: "{user_text}"\n'
        f"Allowed labels: {label_text}\n\n"
        "Assistant:"
    )


def extract_tokens_from_result(result):
    if hasattr(result, "sequences") and result.sequences:
        if hasattr(result.sequences[0], "tokens"):
            return result.sequences[0].tokens

    if hasattr(result, "samples") and result.samples:
        if hasattr(result.samples[0], "tokens"):
            return result.samples[0].tokens

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if "sequences" in dumped and dumped["sequences"]:
            first = dumped["sequences"][0]
            if isinstance(first, dict) and "tokens" in first:
                return first["tokens"]
        if "samples" in dumped and dumped["samples"]:
            first = dumped["samples"][0]
            if isinstance(first, dict) and "tokens" in first:
                return first["tokens"]

    raise ValueError(f"Could not find generated tokens in SampleResponse: {result}")


def run_fine_tuned(user_text, labels, sampling_client, tokenizer):
    prompt_text = build_tinker_prompt(user_text, labels)
    prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(prompt_text))
    sampling_params = tinker.types.SamplingParams(
        max_tokens=8,
        temperature=0.0,
    )

    result = sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    predicted_tokens = extract_tokens_from_result(result)
    raw_prediction = tokenizer.decode(predicted_tokens).strip()
    predicted_label = extract_label(raw_prediction, labels)
    return predicted_label


def run_zero_shot(user_text, labels, model, tokenizer, device):
    messages = build_zero_shot_messages(user_text, labels)
    prompt_text = render_prompt(tokenizer, messages)
    raw_prediction = generate_local_label(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=6,
        device=device,
    )
    return extract_label(raw_prediction, labels)


def run_few_shot(user_text, labels, support_examples, model, tokenizer, device):
    messages = build_few_shot_messages(user_text, labels, support_examples)
    prompt_text = render_prompt(tokenizer, messages)
    raw_prediction = generate_local_label(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=6,
        device=device,
    )
    return extract_label(raw_prediction, labels)


def run_bert(user_text, model, tokenizer, device):
    inputs = tokenizer(
        user_text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

    return model.config.id2label[pred_id]


def main():
    labels = build_label_info()
    train_rows = read_jsonl(ROOT / "data/processed/train.jsonl")
    support_examples = choose_few_shot_examples(train_rows, max_examples=4)

    print(f"Loaded {len(labels)} labels")
    print(f"Using {len(support_examples)} few-shot examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Local device: {device}")

    print("Loading fine-tuned Tinker model...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)
    tinker_tokenizer = sampling_client.get_tokenizer()

    print("Loading base Llama for zero-shot and few-shot...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=True)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_kwargs = {}
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            llama_kwargs["dtype"] = torch.bfloat16
        else:
            llama_kwargs["dtype"] = torch.float16

    llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, **llama_kwargs)
    llama_model.to(device)
    llama_model.eval()

    print("Loading BERT baseline...")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.to(device)
    bert_model.eval()

    print("\nModels ready.\n")

    while True:
        user_text = input("Enter utterance (or type exit): ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_text:
            continue

        fine_tuned_pred = run_fine_tuned(user_text, labels, sampling_client, tinker_tokenizer)
        zero_shot_pred = run_zero_shot(user_text, labels, llama_model, llama_tokenizer, device)
        few_shot_pred = run_few_shot(user_text, labels, support_examples, llama_model, llama_tokenizer, device)
        bert_pred = run_bert(user_text, bert_model, bert_tokenizer, device)

        print("\nResults:")
        print(f"Zero-shot Llama  : {zero_shot_pred}")
        print(f"Few-shot Llama   : {few_shot_pred}")
        print(f"BERT baseline    : {bert_pred}")
        print(f"Fine-tuned Llama : {fine_tuned_pred}\n")


if __name__ == "__main__":
    main()
