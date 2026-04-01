"""Small local inference utilities for label post-processing."""

from __future__ import annotations

from typing import Iterable



def normalize_predicted_label(text: str, allowed_labels: Iterable[str]) -> str:
    """Normalize a raw model output to a canonical intent label when possible.

    The strict project setup asks the model to emit exactly one label. In practice, base models
    may add punctuation or extra words. This helper trims common formatting noise.
    """
    allowed = {label.lower(): label for label in allowed_labels}
    candidate = text.strip().strip('"').strip("'").strip().lower()

    if candidate in allowed:
        return allowed[candidate]

    # Fallback: use the first line and try again.
    first_line = candidate.splitlines()[0].strip()
    if first_line in allowed:
        return allowed[first_line]

    return text.strip()
