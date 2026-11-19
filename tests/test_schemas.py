import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest


def test_predict_request_requires_text():
    with pytest.raises(ValidationError):
        PredictRequest(subject="  ", body="")


def test_predict_request_accepts_subject_only():
    req = PredictRequest(subject="Lab due tomorrow", body="")
    assert req.subject.startswith("Lab")
