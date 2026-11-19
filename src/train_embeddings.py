"""Train a linear head on frozen sentence embeddings (M2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE

MODEL_VERSION = "embeddings-minilm-v1"
DEFAULT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedding_logreg(class_weight: Optional[str] = "balanced") -> Pipeline:
    """Scaler + LogReg on dense vectors. Encoder is applied outside this pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def fit_embedding_classifier(X_train, y_train, class_weight: Optional[str] = "balanced"):
    model = build_embedding_logreg(class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def save_embedding_model(
    model,
    models_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    filename: str = "minilm_logreg.joblib",
) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    joblib.dump(model, path)
    meta = {
        "model_version": MODEL_VERSION,
        "encoder": DEFAULT_ENCODER_NAME,
        "encoder_frozen": True,
        "filename": filename,
        **(metadata or {}),
    }
    (models_dir / "model_meta_m2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
