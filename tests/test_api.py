import numpy as np
from fastapi.testclient import TestClient

from api import index


def test_health_is_unavailable_without_research_model(monkeypatch):
    monkeypatch.setattr(index, "_cached_model", None)
    monkeypatch.setattr(index, "_cached_manifest", None)
    monkeypatch.delenv("NAAD_RESEARCH_USE_ONLY", raising=False)
    with TestClient(index.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unavailable"


def test_predict_does_not_return_fabricated_score_when_unavailable(monkeypatch):
    monkeypatch.setattr(index, "_cached_model", None)
    monkeypatch.setattr(index, "_cached_manifest", None)
    monkeypatch.delenv("NAAD_RESEARCH_USE_ONLY", raising=False)
    with TestClient(index.app) as client:
        response = client.post("/api/predict", json={"text": "I have been feeling low."})
    assert response.status_code == 503
    assert "estimated_phq8_score" not in response.json()


def test_predict_rejects_unknown_audio_field():
    with TestClient(index.app) as client:
        response = client.post("/api/predict", json={"text": "hello", "audio_feats": [0.0] * 32})
    assert response.status_code == 422


def test_predict_returns_explicit_research_only_estimate(monkeypatch):
    class StubModel:
        def predict(self, data):
            assert data.to_dict(orient="records") == [{"transcript": "A participant transcript"}]
            return np.asarray([12.345])

    monkeypatch.setattr(index, "_load_artifacts", lambda: (StubModel(), {"model_version": "test-model"}))
    with TestClient(index.app) as client:
        response = client.post("/api/predict", json={"text": "A participant transcript"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "research_only"
    assert body["estimated_phq8_score"] == 12.35
    assert body["clinical_diagnosis"] is False
    assert body["crisis_assessment"] is False


def test_predict_rejects_nonfinite_model_output(monkeypatch):
    class StubModel:
        def predict(self, data):
            return np.asarray([np.nan])

    monkeypatch.setattr(index, "_load_artifacts", lambda: (StubModel(), {"model_version": "test-model"}))
    with TestClient(index.app) as client:
        response = client.post("/api/predict", json={"text": "A transcript"})
    assert response.status_code == 503
    assert "estimated_phq8_score" not in response.json()
