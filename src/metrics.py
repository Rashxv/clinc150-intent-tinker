"""Metrics helpers for intent classification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



def compute_summary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Compute the main summary metrics requested by the project."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }



def build_per_intent_report(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Create a per-intent precision/recall/F1 table."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    excluded = {"accuracy", "macro avg", "weighted avg"}
    return df[~df["label"].isin(excluded)].sort_values("f1-score", ascending=False)



def build_confusion_df(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Iterable[str],
) -> pd.DataFrame:
    """Create a labeled confusion matrix dataframe."""
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)



def save_metrics_json(metrics: dict, path: Path) -> None:
    """Save a metrics dictionary to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(path, indent=2)
