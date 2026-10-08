import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

app = FastAPI(title="NAAD AI Research API", version="2.0.0")
origins = [value.strip() for value in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("NAAD_MODEL_PATH", ROOT / "models" / "phq8_text_baseline.joblib"))
MANIFEST_PATH = Path(os.getenv("NAAD_MODEL_MANIFEST", ROOT / "models" / "phq8_text_baseline.manifest.json"))
_cached_model = None
_cached_manifest = None


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)

    class Config:
        extra = "forbid"


def _load_artifacts():
    global _cached_model, _cached_manifest
    if _cached_model is not None and _cached_manifest is not None:
        return _cached_model, _cached_manifest
    if os.getenv("NAAD_RESEARCH_USE_ONLY") != "1":
        raise RuntimeError(
            "Research-only model serving is disabled. Enable it only for an authorized research deployment; "
            "this setting does not provide trained model files."
        )
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Research model artifact is missing: {MODEL_PATH.name}")
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"Research model manifest is missing: {MANIFEST_PATH.name}")

    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("task") != "phq8_text_regression" or manifest.get("research_only") is not True:
        raise RuntimeError("Model manifest does not describe the supported research task.")
    if manifest.get("deployable") is not True:
        raise RuntimeError("Model did not pass the documented development baseline gate.")
    if manifest.get("feature_schema") != "tfidf_word_char_ridge_v1":
        raise RuntimeError("Unsupported model feature schema.")
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if model_hash != manifest.get("model_sha256"):
        raise RuntimeError("Model checksum does not match its manifest.")

    # Only load joblib artifacts produced by this repository; joblib uses pickle internally.
    model = joblib.load(MODEL_PATH)
    if not callable(getattr(model, "predict", None)):
        raise RuntimeError("Loaded artifact does not implement predict().")
    _cached_model, _cached_manifest = model, manifest
    return model, manifest


@app.get("/api/health")
def health_check():
    try:
        _load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"status": "unavailable", "reason": str(exc)}) from exc
    return {"status": "ready", "task": "phq8_text_regression", "research_only": True}


@app.post("/api/predict")
def predict(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    try:
        model, manifest = _load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"status": "unavailable", "reason": str(exc)}) from exc

    started = time.perf_counter()
    try:
        # The fitted ColumnTransformer selects the named transcript column.
        predictions = np.asarray(model.predict(pd.DataFrame({"transcript": [text]})), dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research model inference failed; no estimate was produced.") from exc
    if predictions.size != 1 or not np.isfinite(predictions[0]):
        raise HTTPException(status_code=503, detail="Research model returned an invalid estimate.")

    return {
        "status": "research_only",
        "task": "experimental_interview_phq8_estimation",
        "estimated_phq8_score": round(float(np.clip(predictions[0], 0.0, 24.0)), 2),
        "model_version": manifest["model_version"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "clinical_diagnosis": False,
        "crisis_assessment": False,
        "notice": "Research estimate only; not a diagnosis, questionnaire result, or crisis assessment.",
    }


handler = Mangum(app)
