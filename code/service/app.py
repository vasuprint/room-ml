"""FastAPI service that exposes the room classification model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import uuid
import logging

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent  # Go up 2 levels from code/service to project root
MODEL_PATH = PROJECT_ROOT / "code" / "models" / "room_classifier.joblib"
MEETING_ROOMS_PATH = PROJECT_ROOT / "config" / "meeting-rooms.json"
ROOM_MAPPING_PATH = PROJECT_ROOT / "config" / "room-mapping.json"

app = FastAPI(title="Room Recommendation Service", version="2.0.0")

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
                "seats": 35
            }
        }


class RoomRecommendation(BaseModel):
    rank: int
    room: Dict[str, Any]


class RecommendationResponse(BaseModel):
    recommendations: List[RoomRecommendation]
    request_id: str


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache the trained scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python code/train_room_model.py` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_meeting_rooms() -> Dict[str, Any]:
    """Load and cache meeting rooms data."""
    if not MEETING_ROOMS_PATH.exists():
        raise FileNotFoundError(f"Meeting rooms file not found at {MEETING_ROOMS_PATH}")

    with open(MEETING_ROOMS_PATH, 'r', encoding='utf-8') as f:
        rooms_data = json.load(f)

    # Create a dictionary indexed by room_name for fast lookup
    rooms_dict = {room['room_name']: room for room in rooms_data}
    return rooms_dict


@lru_cache(maxsize=1)
def load_room_mapping() -> Dict[str, str]:
    """Load and cache room name mappings."""
    if not ROOM_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Room mapping file not found at {ROOM_MAPPING_PATH}")

    with open(ROOM_MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    return mapping_data['model_to_actual_mapping']


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


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(features: BookingFeatures) -> RecommendationResponse:
    """Return top 3 recommended rooms with details from meeting-rooms.json."""
    try:
        pipeline = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Load room data and mappings
    try:
        meeting_rooms = load_meeting_rooms()
        room_mapping = load_room_mapping()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Configuration file not found: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in configuration file: {str(e)}")

    # Prepare input data
    payload_df = pd.DataFrame([features.dict()])

    # Check if model supports predict_proba
    if not hasattr(pipeline, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail="Current model does not support probability predictions. Please retrain with a compatible model."
        )

    # Get predictions with probabilities
    try:
        proba = pipeline.predict_proba(payload_df)[0]
        classes = pipeline.classes_
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {str(e)}")

    # Sort by probability (descending)
    room_probs = list(zip(classes, proba))
    room_probs.sort(key=lambda x: x[1], reverse=True)

    # Get top 3 recommendations
    recommendations = []
    unmapped_rooms = []
    rank = 1

    for room_name, probability in room_probs[:10]:  # Check top 10 to find at least 3 mappable rooms
        # Map model room name to actual room name
        actual_room_name = room_mapping.get(str(room_name))

        if actual_room_name is None:
            unmapped_rooms.append(str(room_name))
            continue

        if actual_room_name not in meeting_rooms:
            raise HTTPException(
                status_code=500,
                detail=f"Mapped room '{actual_room_name}' not found in meeting-rooms.json for model prediction '{room_name}'"
            )

        room_data = meeting_rooms[actual_room_name]

        recommendations.append(RoomRecommendation(
            rank=rank,
            room={
                "id": room_data["uuid"],
                "name": room_data["room_name"],
                "location": room_data["location_details"],
                "capacity_min": room_data["capacity_min"],
                "capacity_max": room_data["capacity_max"],
                "price": room_data["hourly_rate"]
            }
        ))
        rank += 1

        if len(recommendations) >= 3:
            break

    # Check if we have enough recommendations
    if len(recommendations) == 0:
        raise HTTPException(
            status_code=500,
            detail=f"No room mappings found for predictions: {', '.join(unmapped_rooms[:3])}"
        )

    if len(recommendations) < 3 and unmapped_rooms:
        # Log warning but return what we have
        logger.warning(f"Could not map these rooms: {', '.join(unmapped_rooms)}")

    # Generate request ID
    request_id = f"req-{str(uuid.uuid4())[:8]}"

    return RecommendationResponse(
        recommendations=recommendations,
        request_id=request_id
    )
