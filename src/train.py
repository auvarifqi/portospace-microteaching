"""Train Dummy and TF-IDF + Logistic Regression baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE

MODEL_VERSION = "baseline-v1"


def build_dummy_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )


def build_logreg_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000)),
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


def fit_baselines(
    X_train,
    y_train,
) -> Dict[str, Pipeline]:
    models = {
        "dummy": build_dummy_pipeline(),
        "tfidf_logreg": build_logreg_pipeline(),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def save_model(
    model: Pipeline,
    models_dir: Path,
    metadata: Dict[str, Any] | None = None,
    filename: str = "tfidf_logreg.joblib",
) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    joblib.dump(model, path)
    meta = {
        "model_version": MODEL_VERSION,
        "filename": filename,
        **(metadata or {}),
    }
    meta_path = models_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
