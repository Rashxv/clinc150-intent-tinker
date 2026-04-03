"""Generate predictions from a trained Tinker sampler checkpoint.

Temporary debug-safe version:
- uses the renderer correctly
- inspects the returned SampleResponse shape on the first example
- supports a few common response field names

Usage:
    python scripts/05_generate_predictions.py ^
        --input data/processed/val.jsonl ^
        --sampler-path "tinker://.../sampler_weights/final" ^
        --output results/predictions/val_predictions_200.jsonl ^
        --max-examples 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tinker
from tinker_cookbook import model_info, renderers

from src.dataset_utils import read_jsonl


def normalize_prediction(text: str) -> str:
    """Normalize the model output to a single label string."""
    return str(text).strip().splitlines()[0].strip()




def extract_tokens_from_result(result):
    """Extract generated token ids from the current Tinker SampleResponse shape."""
    if hasattr(result, "sequences"):
        sequences = getattr(result, "sequences")
        if sequences and hasattr(sequences[0], "tokens"):
            return sequences[0].tokens

    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if "sequences" in dumped and dumped["sequences"]:
            first = dumped["sequences"][0]
            if isinstance(first, dict) and "tokens" in first:
                return first["tokens"]

    raise ValueError("Could not find generated tokens in SampleResponse.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sampler-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    if not os.getenv("TINKER_API_KEY"):
        raise EnvironmentError("TINKER_API_KEY not found in environment or .env")

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.sampler_path)

    base_model = sampling_client.get_base_model()
    tokenizer = sampling_client.get_tokenizer()
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            gold_label = row["meta"]["label"]
            utterance = row["meta"]["utterance"]

            # Use the same chat/message format the model saw during training,
            # but without the gold assistant answer.
            messages = [msg for msg in row["messages"] if msg["role"] != "assistant"]

            prompt = renderer.build_generation_prompt(messages)
            stop_sequences = renderer.get_stop_sequences()
            sampling_params = tinker.types.SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.0,
                stop=stop_sequences,
            )

            result = sampling_client.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            

            generated_tokens = extract_tokens_from_result(result)

            # First try renderer-aware parsing
            try:
                sampled_message, parse_success = renderer.parse_response(generated_tokens)
                if parse_success and isinstance(sampled_message, dict):
                    predicted_text = sampled_message.get("content", "")
                elif parse_success and hasattr(sampled_message, "content"):
                    predicted_text = sampled_message.content
                else:
                    predicted_text = tokenizer.decode(generated_tokens)
            except Exception:
                predicted_text = tokenizer.decode(generated_tokens)

            predicted_label = normalize_prediction(predicted_text)

            out_row = {
                "utterance": utterance,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            print(f"[{i}/{len(rows)}] gold={gold_label} pred={predicted_label}")

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()