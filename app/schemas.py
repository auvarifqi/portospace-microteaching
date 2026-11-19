"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    subject: str = Field(default="", max_length=500)
    body: str = Field(default="", max_length=20_000)

    @model_validator(mode="after")
    def require_subject_or_body(self) -> "PredictRequest":
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("Provide a non-empty subject and/or body")
        return self


class PredictResponse(BaseModel):
    label: str
    is_important: bool
    top_terms: List[str] = Field(default_factory=list)
    model_version: str
    note: Optional[str] = (
        "top_terms are model evidence from TF-IDF coefficients, not human causal reasons"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    features: str
    label_mapping: dict
    model_path: str
