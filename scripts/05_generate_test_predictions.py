import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import tinker

sys.path.insert(0, ".")
from src.dataset_utils import read_jsonl

load_dotenv(".env")


def build_label_set():
    labels = set()
    for path in [
        Path("data/processed/train.jsonl"),
        Path("data/processed/val.jsonl"),
        Path("data/processed/test.jsonl"),
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


def build_prompt_from_messages(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "").strip().capitalize()
        content = msg.get("content", "")
        if msg.get("role") == "assistant":
            break
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-file", default="data/processed/test.jsonl")
    parser.add_argument("--output-file", default="results/test_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("Creating service client...")
    service_client = tinker.ServiceClient()

    print("Creating sampling client...")
    sampling_client = service_client.create_sampling_client(model_path=args.model_path)

    print("Loading tokenizer...")
    tokenizer = sampling_client.get_tokenizer()
    print("Tokenizer loaded")

    rows = read_jsonl(Path(args.input_file))
    if args.limit is not None:
        rows = rows[: args.limit]

    all_labels = build_label_set()
    print(f"Loaded {len(rows)} test example(s)")
    print(f"Loaded {len(all_labels)} valid label(s)")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="Generating predictions", unit="example"):
            messages = row.get("messages", [])
            gold_label = row.get("meta", {}).get("label", "")
            utterance = row.get("meta", {}).get("utterance", "")

            prompt_text = build_prompt_from_messages(messages)

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

            if hasattr(result, "sequences") and result.sequences:
                predicted_tokens = result.sequences[0].tokens
            elif hasattr(result, "samples") and result.samples:
                predicted_tokens = result.samples[0].tokens
            else:
                raise RuntimeError(f"Unknown sample response format: {result}")

            predicted_text = tokenizer.decode(predicted_tokens).strip()
            predicted_label = extract_label(predicted_text, all_labels)

            record = {
                "utterance": utterance,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "raw_prediction": predicted_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nDone. Saved predictions to {out_path}")
    print(f"Total time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()