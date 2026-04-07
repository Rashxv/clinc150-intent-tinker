from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import tinker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl

load_dotenv(ROOT / ".env")

MODEL_PATH = "tinker://a347532d-63fe-5f43-ab6d-26cf56c78096:train:0/sampler_weights/final"


def build_label_set():
    labels = set()
    for path in [
        ROOT / "data/processed/train.jsonl",
        ROOT / "data/processed/val.jsonl",
        ROOT / "data/processed/test.jsonl",
    ]:
        if path.exists():
            rows = read_jsonl(path)
            for row in rows:
                label = row.get("meta", {}).get("label", "")
                if label:
                    labels.add(label)
    return sorted(labels, key=len, reverse=True)


def extract_label(predicted_text, all_labels):
    text = predicted_text.strip().lower()

    for label in all_labels:
        if text == label.lower():
            return label

    chunks = re.split(r"[\n,;:]+", text)
    for chunk in chunks:
        chunk = chunk.strip()
        for label in all_labels:
            if chunk == label.lower():
                return label

    for label in all_labels:
        if re.search(rf"\b{re.escape(label.lower())}\b", text):
            return label

    first_line = predicted_text.strip().splitlines()[0].strip() if predicted_text.strip() else ""
    return first_line


def build_prompt(user_text, labels):
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


def main():
    print("Loading labels...")
    labels = build_label_set()
    print(f"Loaded {len(labels)} labels")

    print("Connecting to fine-tuned model...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)

    print("Loading tokenizer...")
    tokenizer = sampling_client.get_tokenizer()
    print("Ready.\n")

    while True:
        user_text = input("Enter utterance (or type exit): ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_text:
            continue

        prompt_text = build_prompt(user_text, labels)
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

        print(f"Predicted intent: {predicted_label}\n")


if __name__ == "__main__":
    main()