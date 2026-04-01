"""Prompt templates for intent classification.

This project frames CLINC150 as instruction tuning:
- the user supplies an utterance
- the assistant outputs exactly one canonical intent label
"""

from __future__ import annotations

from typing import Iterable, List

SYSTEM_PROMPT = (
    "You are an intent classifier. "
    "Output exactly one intent label from the allowed label set. "
    "Do not explain your answer."
)


def build_user_prompt(
    utterance: str,
    labels: Iterable[str],
    include_label_list: bool = True,
) -> str:
    """Build the user prompt used for instruction tuning.

    Args:
        utterance: The raw user utterance.
        labels: Allowed intent labels.
        include_label_list: Whether to include the label vocabulary.

    Returns:
        A user-facing text prompt.
    """
    prompt = f'Utterance: "{utterance.strip()}"'
    if include_label_list:
        label_text = ", ".join(sorted(labels))
        prompt += f"\nAllowed labels: {label_text}"
    return prompt


def build_chat_example(
    utterance: str,
    label: str,
    labels: List[str],
    split: str,
    include_label_list: bool = True,
) -> dict:
    """Create one chat-formatted supervised training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    utterance=utterance,
                    labels=labels,
                    include_label_list=include_label_list,
                ),
            },
            {"role": "assistant", "content": label},
        ],
        "meta": {
            "utterance": utterance,
            "label": label,
            "split": split,
        },
    }
