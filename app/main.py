"""
FastAPI inference service for PriorityLab (M3).

Run after training M1:
  uvicorn app.main:app --reload --port 8000

Requires models/tfidf_logreg.joblib from notebooks/02_baseline_tfidf.ipynb.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.explain import format_term_list, top_terms_for_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "tfidf_logreg.joblib"
DEFAULT_META_PATH = ROOT / "models" / "model_meta.json"

_model: Optional[Pipeline] = None
_model_meta: dict[str, Any] = {}


def _model_path() -> Path:
    return Path(os.getenv("PRIORITYLAB_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def build_text(subject: str, body: str) -> str:
    return f"subject: {subject.strip()} body: {body.strip()}"


def load_artifacts() -> None:
    global _model, _model_meta
    path = _model_path()
    if not path.exists():
        _model = None
        _model_meta = {}
        return
    _model = joblib.load(path)
    meta_path = path.with_name("model_meta.json")
    if not meta_path.exists():
        meta_path = DEFAULT_META_PATH
    if meta_path.exists():
        _model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        _model_meta = {"model_version": "unknown", "filename": path.name}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="PriorityLab",
    description="Explainable email importance classifier API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _model is not None else "model_missing",
        model_loaded=_model is not None,
    )


@app.get("/v1/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train M1 and save models/tfidf_logreg.joblib first.",
        )
    return ModelInfoResponse(
        model_version=str(_model_meta.get("model_version", "unknown")),
        features=str(_model_meta.get("features", "subject+body text")),
        label_mapping={"urgent/high": "important", "medium/low": "not_important"},
        model_path=str(_model_path()),
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train M1 and save models/tfidf_logreg.joblib first.",
        )
    text = build_text(payload.subject, payload.body)
    pred = int(_model.predict([text])[0])
    is_important = pred == 1
    terms = format_term_list(top_terms_for_text(_model, text, k=5))
    return PredictResponse(
        label="important" if is_important else "not_important",
        is_important=is_important,
        top_terms=terms,
        model_version=str(_model_meta.get("model_version", "baseline-v1")),
    )
