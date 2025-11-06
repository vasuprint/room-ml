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
from pydantic import BaseModel, Field, validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping from Thai/English to standard English for event_period
EVENT_PERIOD_MAPPING = {
    # Thai input
    "เช้า": "Morning",
    "บ่าย": "Afternoon",
    "ทั้งวัน": "All Day",
    "กลางคืน": "Night",
    "ค่ำ": "Night",

    # English input (case-insensitive)
    "morning": "Morning",
    "afternoon": "Afternoon",
    "all day": "All Day",
    "allday": "All Day",
    "night": "Night",

    # Standard format (keep as-is)
    "Morning": "Morning",
    "Afternoon": "Afternoon",
    "All Day": "All Day",
    "Night": "Night",
}

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

    @validator('event_period')
    def normalize_event_period(cls, v):
        """Normalize event_period to English (supports both Thai and English input)."""
        if not v:
            raise ValueError("event_period is required")

        # Normalize input: strip whitespace
        normalized = v.strip()

        # Try to map from Thai/English to standard English
        mapped_value = EVENT_PERIOD_MAPPING.get(normalized)

        if mapped_value:
            return mapped_value

        # If not in mapping, raise error with helpful message
        valid_values_thai = ["เช้า", "บ่าย", "ทั้งวัน", "กลางคืน"]
        valid_values_eng = ["Morning", "Afternoon", "All Day", "Night"]
        raise ValueError(
            f"Invalid event_period: '{v}'. "
            f"Accepted Thai values: {valid_values_thai} "
            f"or English values: {valid_values_eng}"
        )

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


def calculate_fit_score(room_data: Dict[str, Any], requested_seats: int) -> float:
    """
    Calculate room suitability based on capacity.

    Args:
        room_data: Room information from meeting-rooms.json
        requested_seats: Number of seats requested

    Returns:
        fit_score (0.0 - 1.0): Higher score means better fit

    Formula:
        fit_score = 1.0 - |requested_seats - capacity_min| / capacity_max

    Example:
        Room capacity: 40-80, Requested: 50
        fit_score = 1.0 - |50-40|/80 = 1.0 - 10/80 = 0.875
    """
    capacity_min = room_data["capacity_min"]
    capacity_max = room_data["capacity_max"]

    # Check if room can accommodate the requested seats
    if requested_seats < capacity_min or requested_seats > capacity_max:
        return 0.0

    # Calculate fitness (closer to capacity_min is better)
    fit_score = 1.0 - abs(requested_seats - capacity_min) / capacity_max

    return fit_score


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
    """Return top 3 recommended rooms based on capacity fit score."""
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

    # =====================================================
    # [OPTIONAL] Get ML predictions for logging/comparison
    # =====================================================
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(payload_df)[0]
            classes = pipeline.classes_
            ml_predictions = list(zip(classes, proba))
            ml_predictions.sort(key=lambda x: x[1], reverse=True)

            # Log ML top 3 for comparison
            ml_top3_names = []
            for room_name, prob in ml_predictions[:3]:
                actual_name = room_mapping.get(str(room_name), str(room_name))
                ml_top3_names.append(f"{actual_name} (prob={prob:.3f})")
            logger.info(f"ML Top 3 predictions: {ml_top3_names}")
    except Exception as e:
        logger.warning(f"Could not get ML predictions: {str(e)}")

    # =====================================================
    # NEW LOGIC: Capacity-based Ranking
    # =====================================================
    requested_seats = features.seats
    suitable_rooms = []

    # Loop through ALL rooms in meeting-rooms.json
    for actual_room_name, room_data in meeting_rooms.items():
        # Calculate fit score (returns 0 if capacity not suitable)
        fit_score = calculate_fit_score(room_data, requested_seats)

        if fit_score > 0:
            suitable_rooms.append({
                'room_name': actual_room_name,
                'room_data': room_data,
                'fit_score': fit_score
            })

    # Sort by fit_score (descending)
    suitable_rooms.sort(key=lambda x: x['fit_score'], reverse=True)

    logger.info(f"Found {len(suitable_rooms)} suitable rooms for {requested_seats} seats")

    # Log top 3 capacity-based recommendations
    if suitable_rooms:
        capacity_top3 = [f"{r['room_name']} (fit={r['fit_score']:.3f})" for r in suitable_rooms[:3]]
        logger.info(f"Capacity-based Top 3: {capacity_top3}")

    # Build recommendations from top 3
    recommendations = []
    for idx, room_info in enumerate(suitable_rooms[:3]):
        room_data = room_info['room_data']

        recommendations.append(RoomRecommendation(
            rank=idx + 1,
            room={
                "id": room_data["uuid"],
                "name": room_data["room_name"],
                "location": room_data["location_details"],
                "capacity_min": room_data["capacity_min"],
                "capacity_max": room_data["capacity_max"],
                "price": room_data["hourly_rate"],
                "fit_score": round(room_info['fit_score'], 3)  # Add fit_score to response
            }
        ))

    # Handle case: no suitable rooms found
    if len(recommendations) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No suitable rooms found for {requested_seats} seats. "
                   f"Available capacity range: 1-700 seats."
        )

    # Generate request ID
    request_id = f"req-{str(uuid.uuid4())[:8]}"

    return RecommendationResponse(
        recommendations=recommendations,
        request_id=request_id
    )
