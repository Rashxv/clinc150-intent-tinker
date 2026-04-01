"""Placeholder evaluator hooks for Tinker runs.

This file intentionally stays lightweight. The exact evaluator-builder integration can vary
with the cookbook version you install. Keep your local metrics logic here, and wire it into
Tinker's inline or offline evaluation once your environment is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.metrics import compute_summary_metrics


@dataclass
class EvalBatch:
    """Simple container for predicted and gold labels."""

    y_true: Sequence[str]
    y_pred: Sequence[str]



def evaluate_predictions(batch: EvalBatch) -> dict:
    """Evaluate a batch of predictions with project metrics."""
    return compute_summary_metrics(batch.y_true, batch.y_pred)
