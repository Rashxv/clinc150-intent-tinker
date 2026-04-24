"""Gradio demo app — CLINC150 Intent Classification.

Launch:
    HF_TOKEN=<your_token> python scripts/demo_app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import os
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
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import read_jsonl

load_dotenv(ROOT / ".env")

# ── Model paths ────────────────────────────────────────────────────────────────
TINKER_MODEL_PATH = "tinker://6cc3a560-3451-582c-8baf-dca20d7a7dff:train:0/sampler_weights/final"
LLAMA_MODEL_NAME  = "meta-llama/Llama-3.2-1B-Instruct"
BERT_MODEL_PATH   = ROOT / "results" / "bert_baseline" / "best_model"

EXAMPLES = [
    "what's my card balance?",
    "remind me to take my medicine at 8pm",
    "how do you say goodbye in Japanese?",
    "what's the weather like in Dubai?",
    "play some jazz music",
    "can I make a reservation at a restaurant?",
    "I need to report my card as lost",
    "transfer $200 to my savings account",
    "set an alarm for 6am tomorrow",
    "what can you help me with?",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_labels() -> list[str]:
    labels: set[str] = set()
    for path in [
        ROOT / "data/processed/train.jsonl",
        ROOT / "data/processed/val.jsonl",
        ROOT / "data/processed/test.jsonl",
    ]:
        if path.exists():
            for row in read_jsonl(path):
                label = row.get("meta", {}).get("label", "").strip()
                if label:
                    labels.add(label)
    return sorted(labels)


def group_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("meta", {}).get("label", "").strip()
        if label:
            grouped[label].append(row)
    return grouped


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


def choose_few_shot_examples(train_rows: list[dict], n: int = 4) -> list[dict]:
    grouped = group_by_label(train_rows)
    labels = sorted(grouped.keys())
    support, idx = [], 0
    while len(support) < n and labels:
        label = labels[idx % len(labels)]
        take = idx // len(labels)
        if take < len(grouped[label]):
            support.append(grouped[label][take])
        idx += 1
        if idx > len(labels) * max(len(v) for v in grouped.values()):
            break
    return support[:n]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_label(predicted_text: str, all_labels: list[str]) -> str:
    text = predicted_text.strip()
    norm = normalize(text)
    for label in all_labels:
        if norm == label.lower():
            return label
    cleaned = text.strip('"\'').strip()
    for label in all_labels:
        if normalize(cleaned) == label.lower():
            return label
    chunks = re.split(r"[\n,;:]+", text)
    for chunk in chunks:
        c = normalize(chunk.strip('"\'').strip())
        for label in all_labels:
            if c == label.lower():
                return label
    for label in sorted(all_labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label.lower())}\b", norm):
            return label
    return text.splitlines()[0].strip() if text else ""


# ── Model loading ──────────────────────────────────────────────────────────────

print("Loading labels and few-shot examples...")
ALL_LABELS = read_labels()
TRAIN_ROWS = read_jsonl(ROOT / "data/processed/train.jsonl")
SUPPORT_EXAMPLES = choose_few_shot_examples(TRAIN_ROWS, n=4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Labels: {len(ALL_LABELS)} | Few-shot examples: {len(SUPPORT_EXAMPLES)}")

print("Loading Fine-tuned Tinker model...")
_service  = tinker.ServiceClient()
_sampling = _service.create_sampling_client(model_path=TINKER_MODEL_PATH)
TINKER_TOKENIZER = _sampling.get_tokenizer()
SAMPLING_CLIENT  = _sampling

print("Loading Llama-3.2-1B-Instruct...")
_llama_kwargs = {}
if torch.cuda.is_available():
    _llama_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=True)
if LLAMA_TOKENIZER.pad_token is None:
    LLAMA_TOKENIZER.pad_token = LLAMA_TOKENIZER.eos_token
LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME, **_llama_kwargs)
LLAMA_MODEL.to(DEVICE).eval()

print("Loading BERT baseline...")
BERT_TOKENIZER = AutoTokenizer.from_pretrained(str(BERT_MODEL_PATH))
BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_PATH))
BERT_MODEL.to(DEVICE).eval()

print("All models ready.\n")


# ── Inference functions ────────────────────────────────────────────────────────

def run_bert(utterance: str) -> str:
    inputs = BERT_TOKENIZER(utterance, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = BERT_MODEL(**inputs).logits
    return BERT_MODEL.config.id2label[int(torch.argmax(logits, dim=-1).item())]


def run_llama(messages: list[dict], max_new_tokens: int = 8) -> str:
    prompt = LLAMA_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = LLAMA_TOKENIZER(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        out = LLAMA_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=LLAMA_TOKENIZER.eos_token_id,
            eos_token_id=LLAMA_TOKENIZER.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return LLAMA_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


def run_zero_shot(utterance: str) -> str:
    label_text = ", ".join(ALL_LABELS)
    messages = [
        {"role": "system", "content": (
            "You are an intent classifier. "
            "Choose exactly one label from the allowed labels. "
            "Reply with exactly one label. Do not explain.\n"
            f"Allowed labels: {label_text}"
        )},
        {"role": "user", "content": f'Utterance: "{utterance}"'},
    ]
    raw = run_llama(messages)
    return extract_label(raw, ALL_LABELS)


def run_few_shot(utterance: str) -> str:
    label_text = ", ".join(ALL_LABELS)
    messages = [
        {"role": "system", "content": (
            "You are an intent classifier. "
            "Choose exactly one label from the allowed labels. "
            "Reply with exactly one label. Do not explain.\n"
            f"Allowed labels: {label_text}"
        )},
    ]
    for ex in SUPPORT_EXAMPLES:
        messages.append({"role": "user",      "content": f'Utterance: "{get_utterance(ex)}"'})
        messages.append({"role": "assistant", "content": ex.get("meta", {}).get("label", "")})
    messages.append({"role": "user", "content": f'Utterance: "{utterance}"'})
    raw = run_llama(messages)
    return extract_label(raw, ALL_LABELS)


def run_fine_tuned(utterance: str) -> str:
    label_text = ", ".join(ALL_LABELS)
    prompt = (
        "System: You are an intent classifier. "
        "Choose exactly one label from the allowed labels. "
        "Output only the label and nothing else.\n\n"
        f'User: Utterance: "{utterance}"\n'
        f"Allowed labels: {label_text}\n\n"
        "Assistant:"
    )
    token_ids = TINKER_TOKENIZER.encode(prompt)
    model_input = tinker.types.ModelInput.from_ints(token_ids)
    result = SAMPLING_CLIENT.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(max_tokens=8, temperature=0.0),
    ).result()

    if hasattr(result, "sequences") and result.sequences:
        tokens = result.sequences[0].tokens
    elif hasattr(result, "samples") and result.samples:
        tokens = result.samples[0].tokens
    else:
        raise RuntimeError(f"Unexpected Tinker response format: {result}")

    raw = TINKER_TOKENIZER.decode(tokens).strip()
    return extract_label(raw, ALL_LABELS)


# ── Gradio inference handler ───────────────────────────────────────────────────

def classify(utterance: str):
    utterance = utterance.strip()
    if not utterance:
        return "—", "—", "—", "—", ""

    bert_pred   = run_bert(utterance)
    zero_pred   = run_zero_shot(utterance)
    few_pred    = run_few_shot(utterance)
    ft_pred     = run_fine_tuned(utterance)

    preds = [bert_pred, zero_pred, few_pred, ft_pred]
    agreement = len(set(preds))
    if agreement == 1:
        note = "✅ All 4 models agree"
    elif agreement == 2:
        note = "⚠️ Models partially disagree"
    else:
        note = "❌ Models mostly disagree"

    return bert_pred, zero_pred, few_pred, ft_pred, note


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="CLINC150 Intent Classification Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎯 CLINC150 Intent Classification Demo
    **Compare 4 approaches** — type any utterance and see how each model classifies it.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            utterance_box = gr.Textbox(
                label="User Utterance",
                placeholder='e.g. "what is my card balance?"',
                lines=2,
            )
            classify_btn = gr.Button("Classify", variant="primary", size="lg")

            gr.Examples(
                examples=EXAMPLES,
                inputs=utterance_box,
                label="Quick examples — click to load",
            )

        with gr.Column(scale=3):
            gr.Markdown("### Results")
            with gr.Row():
                bert_out = gr.Textbox(label="🔵 BERT (fine-tuned encoder)", interactive=False)
                zero_out = gr.Textbox(label="🟡 Llama zero-shot",           interactive=False)
            with gr.Row():
                few_out  = gr.Textbox(label="🟠 Llama few-shot",            interactive=False)
                ft_out   = gr.Textbox(label="🟢 Fine-tuned Llama (LoRA)",   interactive=False)

            agreement_box = gr.Markdown("")

    gr.Markdown("""
    ---
    | Model | Training data used | Approach |
    |---|---|---|
    | 🔵 BERT | 15,000 labeled examples | Full fine-tune of encoder |
    | 🟡 Llama zero-shot | None | Prompt only — list all 150 labels |
    | 🟠 Llama few-shot | 4 examples in prompt | Prompt with 4 demonstrations |
    | 🟢 Fine-tuned Llama | 15,000 labeled examples | LoRA SFT via Tinker |
    """)

    classify_btn.click(
        fn=classify,
        inputs=utterance_box,
        outputs=[bert_out, zero_out, few_out, ft_out, agreement_box],
    )
    utterance_box.submit(
        fn=classify,
        inputs=utterance_box,
        outputs=[bert_out, zero_out, few_out, ft_out, agreement_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
