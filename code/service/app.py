"""FastAPI service that exposes the room classification model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "room_classifier.joblib"

app = FastAPI(title="Room Recommendation Service", version="1.0.0")

# Allow the Vue dev server to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # change this to Vue dev server address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BookingFeatures(BaseModel):
    department: str = Field(..., example="งานบริหารทั่วไป")
    duration_hours: float = Field(..., ge=0.0, example=2.5)
    event_period: str = Field(..., example="บ่าย")
    seats: int = Field(..., ge=1, example=40)

    class Config:
        schema_extra = {
            "example": {
                "department": "งานบริหารทั่วไป",
                "duration_hours": 2.0,
                "event_period": "บ่าย",
                "seats": 35,
            }
        }


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache the trained scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python code/train_room_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def _warm_model() -> None:
    """Ensure the model is loaded when the service starts."""
    try:
        load_model()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    """Simple readiness probe for deployments."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: BookingFeatures) -> Dict[str, str]:
    """Return the recommended room for the provided booking features."""
    pipeline = load_model()

    payload_df = pd.DataFrame([features.dict()])

    try:
        prediction = pipeline.predict(payload_df)[0]
    except Exception as exc:  # pragma: no cover - scikit errors surface here
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"room": str(prediction)}


@app.post("/predict_proba")
def predict_proba(features: BookingFeatures) -> Dict[str, Dict[str, float]]:
    """Optional endpoint to expose class probabilities when supported."""
    pipeline = load_model()

    if not hasattr(pipeline, "predict_proba"):
        raise HTTPException(status_code=400, detail="Model does not support predict_proba.")

    payload_df = pd.DataFrame([features.dict()])

    proba = pipeline.predict_proba(payload_df)[0]
    classes = pipeline.classes_
    return {
        "probabilities": {
            str(room): float(prob) for room, prob in zip(classes, proba)
        }
    }
