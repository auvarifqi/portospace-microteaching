"""API smoke tests — work without a trained model (expect 503 on predict)."""

from fastapi.testclient import TestClient

from app.main import app, load_artifacts


def test_health_endpoint():
    load_artifacts()
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_validation_empty():
    client = TestClient(app)
    res = client.post("/v1/predict", json={"subject": "", "body": ""})
    assert res.status_code == 422


def test_predict_without_model_is_503_or_ok():
    """If joblib exists locally this may return 200; otherwise 503."""
    load_artifacts()
    client = TestClient(app)
    res = client.post(
        "/v1/predict",
        json={"subject": "Report due tomorrow", "body": "Please submit by 08:00"},
    )
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        body = res.json()
        assert body["label"] in ("important", "not_important")
        assert "model_version" in body
