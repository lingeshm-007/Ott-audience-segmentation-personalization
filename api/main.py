"""
FastAPI Audience Segmentation & Personalization Service.
Exposes /health and /recommend endpoints powered by persisted KMeans and Preprocessing artifacts.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Ensure API directory and trainer modules are in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
TRAINER_DIR = os.path.join(os.path.dirname(BASE_DIR), "trainer")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

from preprocessing import OTTFeaturePreprocessor, GENRE_LIST, NUMERICAL_FEATURES
from recommendations import generate_recommendations

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ott-api")

# Global state for loaded model artifacts
MODEL_STATE: Dict[str, Any] = {
    "preprocessor": None,
    "kmeans_model": None,
    "segment_metadata": {},
    "model_loaded": False,
}

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(BASE_DIR), "models"))


def load_artifacts():
    """Load persisted preprocessor, model, and metadata into memory."""
    try:
        prep_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
        model_path = os.path.join(MODEL_DIR, "kmeans_model.joblib")
        meta_path = os.path.join(MODEL_DIR, "segment_metadata.json")

        if not (os.path.exists(prep_path) and os.path.exists(model_path) and os.path.exists(meta_path)):
            logger.warning(f"Model artifacts not completely found in: {MODEL_DIR}")
            MODEL_STATE["model_loaded"] = False
            return

        MODEL_STATE["preprocessor"] = joblib.load(prep_path)
        MODEL_STATE["kmeans_model"] = joblib.load(model_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            MODEL_STATE["segment_metadata"] = json.load(f)

        MODEL_STATE["model_loaded"] = True
        logger.info(f"Model artifacts successfully loaded from: {MODEL_DIR}")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        MODEL_STATE["model_loaded"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="OTT Audience Segmentation & Personalization API",
    description="Containerized REST API for clustering audience behavior and delivering personalized recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request & Response Schemas ---

class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", min_length=1)
    watch_time_hours: float = Field(..., description="Total watch time in hours", ge=0.0, le=1000.0)
    avg_session_mins: float = Field(..., description="Average session length in minutes", ge=0.0, le=1440.0)
    top_genres: List[str] = Field(default_factory=list, description="List of user's favorite genres")
    
    # Optional extended behavioral features
    sessions_per_week: Optional[float] = Field(None, description="Average sessions per week", ge=0.0, le=100.0)
    completion_rate: Optional[float] = Field(None, description="Average video completion rate (0.0 - 1.0)", ge=0.0, le=1.0)
    weekend_usage_ratio: Optional[float] = Field(None, description="Fraction of viewing during weekends (0.0 - 1.0)", ge=0.0, le=1.0)
    genre_diversity: Optional[float] = Field(None, description="Number of distinct genres explored", ge=0.0, le=20.0)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_id must not be empty or whitespace only")
        return v.strip()

    @field_validator("top_genres")
    @classmethod
    def sanitize_genres(cls, v: List[str]) -> List[str]:
        # Strip whitespace and ignore empty entries
        return [str(item).strip() for item in v if item is not None and str(item).strip()]


class RecommendResponse(BaseModel):
    user_id: str
    segment_id: int
    segment_name: str
    recommendations: List[str]
    distance_to_centroid: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- Exception Handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error: " + "; ".join(errors)},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {str(exc)}", exc_info=False)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred processing request."},
    )


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to inspect service and model status."""
    is_loaded = MODEL_STATE["model_loaded"]
    return {
        "status": "ok" if is_loaded else "unhealthy",
        "model_loaded": is_loaded,
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest):
    """
    Perform inference to segment user and provide personalized recommendations.
    Uses purely persisted model artifacts — no retraining occurs.
    """
    if not MODEL_STATE["model_loaded"]:
        # Attempt runtime reload in case models were written after startup
        load_artifacts()
        if not MODEL_STATE["model_loaded"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts are not loaded. Run trainer first.",
            )

    preprocessor: OTTFeaturePreprocessor = MODEL_STATE["preprocessor"]
    kmeans_model = MODEL_STATE["kmeans_model"]
    segment_metadata = MODEL_STATE["segment_metadata"]

    # Sanitize genres
    sanitized_genres = [g for g in payload.top_genres if g in GENRE_LIST]

    # Impute or infer optional features if not explicitly supplied
    if payload.sessions_per_week is not None:
        sess_per_wk = payload.sessions_per_week
    else:
        # Heuristic estimation: (watch_time_hours * 60 / session_mins) / 4 weeks
        if payload.avg_session_mins > 0:
            estimated = (payload.watch_time_hours * 60.0) / payload.avg_session_mins / 4.0
            sess_per_wk = max(1.0, min(14.0, estimated))
        else:
            sess_per_wk = 3.0

    completion_rate = payload.completion_rate if payload.completion_rate is not None else preprocessor.median_completion_rate
    weekend_ratio = payload.weekend_usage_ratio if payload.weekend_usage_ratio is not None else 0.48
    diversity = payload.genre_diversity if payload.genre_diversity is not None else max(1.0, float(len(sanitized_genres)))

    # Format input row for preprocessor
    row_df = pd.DataFrame([{
        "user_id": payload.user_id,
        "watch_time_hours": payload.watch_time_hours,
        "avg_session_mins": payload.avg_session_mins,
        "sessions_per_week": sess_per_wk,
        "completion_rate": completion_rate,
        "weekend_usage_ratio": weekend_ratio,
        "genre_diversity": diversity,
        "top_genres": "|".join(payload.top_genres),
    }])

    # Preprocess & predict using persisted KMeans model (NO RETRAINING)
    X_sample = preprocessor.transform(row_df)
    predicted_cluster = int(kmeans_model.predict(X_sample)[0])
    
    # Calculate Euclidean distance to cluster centroid
    all_distances = kmeans_model.transform(X_sample)
    distance_to_centroid = float(all_distances[0][predicted_cluster])

    meta = segment_metadata.get(str(predicted_cluster), {})
    segment_name = meta.get("segment_name", f"Segment-{predicted_cluster}")
    rec_genres = meta.get("recommended_genres", ["Drama", "Comedy"])

    # Generate rule-based personalized recommendations
    recs = generate_recommendations(
        segment_name=segment_name,
        recommended_genres=rec_genres,
        user_top_genres=payload.top_genres,
        limit=4,
    )

    return {
        "user_id": payload.user_id,
        "segment_id": predicted_cluster,
        "segment_name": segment_name,
        "recommendations": recs,
        "distance_to_centroid": round(distance_to_centroid, 4),
    }
