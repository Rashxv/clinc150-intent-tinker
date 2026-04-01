# CLINC150 Intent Classification with Tinker

Undergraduate LLM fine-tuning project scaffold for **Intent Classification for Conversational Queries** using the **CLINC150** dataset and **Tinker**.

## Project goal

Fine-tune a language model to classify short user utterances into one of the CLINC150 intent labels, then evaluate the model with:

- accuracy
- macro-F1
- per-intent precision / recall / F1
- confusion analysis
- train / validation loss curves

This repo is organized to match the course rubric:

1. dataset preparation
2. validation-driven hyperparameter tuning
3. loss-function setup
4. training and validation plots
5. final test evaluation and demo-ready inference

## Recommended project scope

### Phase 1 — core project
- Train on **in-scope intents only**
- Output exactly **one canonical intent label** per utterance
- Compare fine-tuned model against a base-model baseline

### Phase 2 — extension
- Add **out-of-scope (OOS)** support
- Compare with and without OOS handling
- Analyze failure modes on ambiguous intents

## Repository layout

```text
clinc150-intent-tinker/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── sweep_lr.yaml
│   └── final_run.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── scripts/
│   ├── 01_load_dataset.py
│   ├── 02_prepare_data.py
│   ├── 03_baseline_eval.py
│   ├── 04_train_tinker.py
│   ├── 05_eval_test.py
│   ├── 06_confusion_analysis.py
│   └── 07_plot_curves.py
├── src/
│   ├── __init__.py
│   ├── prompts.py
│   ├── dataset_utils.py
│   ├── metrics.py
│   ├── evaluators.py
│   └── inference.py
├── results/
├── notebooks/
└── report/
```

## Quick start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
```

### 2) Add your Tinker key

Copy `.env.example` to `.env` and fill in your key.

```bash
cp .env.example .env
```

### 3) Download CLINC150

```bash
python scripts/01_load_dataset.py
```

### 4) Build processed instruction-response data

```bash
python scripts/02_prepare_data.py --include-oos false
```

### 5) Run a simple baseline

```bash
python scripts/03_baseline_eval.py --split val
```

### 6) Launch a Tinker training run

```bash
python scripts/04_train_tinker.py --config configs/base.yaml
```

### 7) Evaluate on test set

```bash
python scripts/05_eval_test.py --predictions results/runs/latest/test_predictions.jsonl
```

## Data format used for Tinker SFT

Each example is converted to a chat-style instruction / response pair.

### Example prompt

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an intent classifier. Output exactly one intent label from the allowed label set. Do not explain your answer."
    },
    {
      "role": "user",
      "content": "Utterance: \"what is my card balance\"\nAllowed labels: [abbreviated list or external metadata]"
    },
    {
      "role": "assistant",
      "content": "balance"
    }
  ],
  "meta": {
    "utterance": "what is my card balance",
    "label": "balance",
    "split": "train"
  }
}
```

## What this scaffold does and does not do

### Included
- dataset loading from Hugging Face
- intent-label metadata export
- instruction-response conversion
- local metrics and confusion analysis
- plotting utilities
- **honest Tinker training skeleton** with clear TODO markers

### Not hard-coded
- account-specific Tinker job settings
- exact model choice
- exact evaluator builder classes from your chosen cookbook version

That part is left explicit on purpose so the scaffold stays accurate and easy to adapt.

## Suggested experiments

1. **Base model zero-shot**
2. **Base model few-shot**
3. **Fine-tuned model (LoRA)**
4. **Learning-rate sweep on validation set**
5. **With vs without OOS**

## Deliverables checklist

- [ ] processed dataset with documentation
- [ ] baseline results
- [ ] fine-tuned model on Tinker
- [ ] accuracy + macro-F1
- [ ] per-intent table
- [ ] confusion matrix
- [ ] train / validation loss plots
- [ ] short technical report
- [ ] live demo on test-style utterances

