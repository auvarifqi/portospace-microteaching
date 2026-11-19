"""Evaluation helpers — compute metrics only from a fitted model + held-out data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

COLORS = {
    "blue": "#4285F4",
    "red": "#EA4335",
    "yellow": "#FBBC04",
    "green": "#34A853",
}


def classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_important": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_important": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_important": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "support": {
            "important": int(np.sum(np.asarray(y_true) == 1)),
            "not_important": int(np.sum(np.asarray(y_true) == 0)),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["not_important", "important"],
            zero_division=0,
        ),
    }


def evaluate_model(model, X, y) -> Dict[str, Any]:
    y_pred = model.predict(X)
    return classification_metrics(y, y_pred)


def plot_confusion_matrix(
    cm,
    out_path: Path,
    title: str = "Confusion matrix",
) -> Path:
    """cm expected as [[tn, fp], [fn, tp]] for labels [0, 1]."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metrics(metrics: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep report text; drop nothing required for portfolio honesty.
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
