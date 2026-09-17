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
from fastapi.responses import JSONResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def interactive_browser_ui():
    """Interactive browser dashboard for audience segmentation and personalized recommendations."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OTT Audience Segmentation & Personalization</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f293d;
      --accent: #e50914;
      --accent-hover: #f40612;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --badge-bg: #1e293b;
      --card-highlight: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      line-height: 1.5;
    }
    header {
      background: linear-gradient(180deg, rgba(17,24,39,0.95) 0%, rgba(9,13,22,0.8) 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      backdrop-filter: blur(12px);
      z-index: 10;
    }
    .logo {
      font-weight: 800;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .logo span { color: var(--accent); }
    .nav-links { display: flex; gap: 1rem; align-items: center; }
    .nav-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.875rem;
      font-weight: 500;
      transition: color 0.2s;
    }
    .nav-links a:hover { color: var(--text); }
    .status-badge {
      font-size: 0.75rem;
      font-weight: 600;
      background: #064e3b;
      color: #34d399;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .status-dot { width: 6px; height: 6px; background: #34d399; border-radius: 50%; }

    main {
      flex: 1;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2rem;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.75rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .panel-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .preset-btns {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .btn-preset {
      background: var(--badge-bg);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.4rem 0.75rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-preset:hover, .btn-preset.active {
      background: #374151;
      color: #fff;
      border-color: #4b5563;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    .form-group label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .form-input {
      background: #0b0f19;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      color: var(--text);
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }
    .form-input:focus { border-color: var(--accent); }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

    .genre-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }
    .genre-chip {
      background: #0b0f19;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 0.35rem 0.7rem;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
      user-select: none;
      transition: all 0.2s;
    }
    .genre-chip.selected {
      background: rgba(229, 9, 20, 0.2);
      border-color: var(--accent);
      color: #ff8080;
      font-weight: 600;
    }
    .btn-submit {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      margin-top: 0.5rem;
      transition: background 0.2s, transform 0.1s;
    }
    .btn-submit:hover { background: var(--accent-hover); }
    .btn-submit:active { transform: scale(0.99); }

    /* Result Panel */
    .result-card {
      background: #0b0f19;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .seg-badge {
      align-self: flex-start;
      background: rgba(229, 9, 20, 0.15);
      border: 1px solid var(--accent);
      color: #ff6b6b;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      text-transform: uppercase;
    }
    .seg-name {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: -0.01em;
    }
    .seg-desc {
      font-size: 0.9rem;
      color: var(--text-muted);
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      border-top: 1px solid var(--card-border);
      padding-top: 0.75rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .meta-row strong { color: var(--text); }

    .rec-title {
      font-size: 0.95rem;
      font-weight: 700;
      margin-top: 0.5rem;
    }
    .rec-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .movie-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      position: relative;
      overflow: hidden;
    }
    .movie-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--accent), #ff8080);
    }
    .movie-name { font-weight: 700; font-size: 0.95rem; }
    .movie-tag { font-size: 0.75rem; color: #a78bfa; font-weight: 600; }

    footer {
      text-align: center;
      padding: 1.5rem;
      color: var(--text-muted);
      font-size: 0.8rem;
      border-top: 1px solid var(--card-border);
    }
  </style>
</head>
<body>

  <header>
    <div class="logo">
      <span>OTT</span> Audience Segmentation
    </div>
    <div class="nav-links">
      <div class="status-badge"><div class="status-dot"></div> Model Live (K=5)</div>
      <a href="/docs" target="_blank">Swagger API</a>
      <a href="/health" target="_blank">/health</a>
    </div>
  </header>

  <main>
    <!-- Left: Profile Input -->
    <div class="panel">
      <div class="panel-title">👤 Viewer Telemetry Profile</div>

      <div class="form-group">
        <label>Quick Archetype Presets:</label>
        <div class="preset-btns">
          <button class="btn-preset active" onclick="loadPreset('power')">Power Action/Sci-Fi</button>
          <button class="btn-preset" onclick="loadPreset('casual')">Casual Weekend</button>
          <button class="btn-preset" onclick="loadPreset('explorer')">Genre Explorer</button>
          <button class="btn-preset" onclick="loadPreset('drama')">Focused Drama</button>
          <button class="btn-preset" onclick="loadPreset('family')">Family & Kids</button>
        </div>
      </div>

      <div class="grid-2">
        <div class="form-group">
          <label>User ID</label>
          <input id="userId" class="form-input" value="USR-8192" />
        </div>
        <div class="form-group">
          <label>Watch Time (Hours/Mo)</label>
          <input id="watchTime" class="form-input" type="number" step="0.5" value="45.0" />
        </div>
      </div>

      <div class="grid-2">
        <div class="form-group">
          <label>Avg Session (Mins)</label>
          <input id="sessionMins" class="form-input" type="number" step="1" value="95.0" />
        </div>
        <div class="form-group">
          <label>Sessions / Week</label>
          <input id="sessWeek" class="form-input" type="number" step="1" value="11" />
        </div>
      </div>

      <div class="grid-2">
        <div class="form-group">
          <label>Completion Rate (0.0 - 1.0)</label>
          <input id="compRate" class="form-input" type="number" step="0.05" min="0" max="1" value="0.85" />
        </div>
        <div class="form-group">
          <label>Weekend Usage Ratio (0.0 - 1.0)</label>
          <input id="weekendRatio" class="form-input" type="number" step="0.05" min="0" max="1" value="0.35" />
        </div>
      </div>

      <div class="form-group">
        <label>Top Genres (Click to toggle):</label>
        <div class="genre-grid" id="genreContainer"></div>
      </div>

      <button class="btn-submit" onclick="runInference()">Segment User & Get Recommendations</button>
    </div>

    <!-- Right: Result & Recommendations -->
    <div class="panel">
      <div class="panel-title">🎯 Model Output & Personalization</div>

      <div id="outputContainer" style="display: flex; flex-direction: column; gap: 1rem;">
        <div class="result-card">
          <div class="seg-badge" id="segBadge">SEGMENT 0</div>
          <div class="seg-name" id="segName">Heavy Action & Sci-Fi Power Viewers</div>
          <div class="seg-desc" id="segDesc">High watch time, high frequency, long sessions; strong affinity for Action, Sci-Fi, and Thriller.</div>
          
          <div class="meta-row">
            <div>User ID: <strong id="resUserId">USR-8192</strong></div>
            <div>Distance to Centroid: <strong id="resDist">1.5673</strong></div>
          </div>
        </div>

        <div class="rec-title">✨ Personalized Recommendations for this Profile:</div>
        <div class="rec-grid" id="recGrid">
          <!-- Movies rendered dynamically -->
        </div>
      </div>
    </div>
  </main>

  <footer>
    Containerized Audience Segmentation & Personalization Service • Powered by Scikit-Learn KMeans & FastAPI
  </footer>

  <script>
    const ALL_GENRES = ["Action", "Animation", "Comedy", "Documentary", "Drama", "Family", "Romance", "Sci-Fi", "Thriller"];
    let selectedGenres = new Set(["Action", "Sci-Fi", "Thriller"]);

    const PRESETS = {
      power: { id: "USR-POWER-1", watch: 48.0, session: 95.0, sessWk: 11, comp: 0.85, weekend: 0.35, genres: ["Action", "Sci-Fi", "Thriller"] },
      casual: { id: "USR-CASUAL-2", watch: 5.0, session: 35.0, sessWk: 2, comp: 0.50, weekend: 0.75, genres: ["Comedy", "Drama"] },
      explorer: { id: "USR-EXPLORE-3", watch: 30.0, session: 60.0, sessWk: 7, comp: 0.68, weekend: 0.40, genres: ["Romance", "Thriller", "Sci-Fi", "Documentary"] },
      drama: { id: "USR-DRAMA-4", watch: 27.0, session: 75.0, sessWk: 6, comp: 0.82, weekend: 0.40, genres: ["Drama", "Romance", "Thriller"] },
      family: { id: "USR-FAMILY-5", watch: 11.0, session: 26.0, sessWk: 4, comp: 0.63, weekend: 0.52, genres: ["Animation", "Family", "Comedy"] }
    };

    function renderGenres() {
      const container = document.getElementById('genreContainer');
      container.innerHTML = '';
      ALL_GENRES.forEach(g => {
        const chip = document.createElement('div');
        chip.className = 'genre-chip' + (selectedGenres.has(g) ? ' selected' : '');
        chip.innerText = g;
        chip.onclick = () => {
          if (selectedGenres.has(g)) selectedGenres.delete(g);
          else selectedGenres.add(g);
          renderGenres();
        };
        container.appendChild(chip);
      });
    }

    function loadPreset(key) {
      document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      const p = PRESETS[key];
      document.getElementById('userId').value = p.id;
      document.getElementById('watchTime').value = p.watch;
      document.getElementById('sessionMins').value = p.session;
      document.getElementById('sessWeek').value = p.sessWk;
      document.getElementById('compRate').value = p.comp;
      document.getElementById('weekendRatio').value = p.weekend;
      selectedGenres = new Set(p.genres);
      renderGenres();
      runInference();
    }

    async function runInference() {
      const payload = {
        user_id: document.getElementById('userId').value || 'USR-TEST',
        watch_time_hours: parseFloat(document.getElementById('watchTime').value) || 0.0,
        avg_session_mins: parseFloat(document.getElementById('sessionMins').value) || 0.0,
        sessions_per_week: parseFloat(document.getElementById('sessWeek').value) || 3.0,
        completion_rate: parseFloat(document.getElementById('compRate').value) || 0.69,
        weekend_usage_ratio: parseFloat(document.getElementById('weekendRatio').value) || 0.48,
        genre_diversity: selectedGenres.size,
        top_genres: Array.from(selectedGenres)
      };

      try {
        const res = await fetch('/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        document.getElementById('segBadge').innerText = 'SEGMENT ' + data.segment_id;
        document.getElementById('segName').innerText = data.segment_name;
        document.getElementById('resUserId').innerText = data.user_id;
        document.getElementById('resDist').innerText = data.distance_to_centroid.toFixed(4);

        const recGrid = document.getElementById('recGrid');
        recGrid.innerHTML = '';
        (data.recommendations || []).forEach(title => {
          const card = document.createElement('div');
          card.className = 'movie-card';
          card.innerHTML = `<div class="movie-name">${title}</div><div class="movie-tag">Tailored Match</div>`;
          recGrid.appendChild(card);
        });
      } catch (err) {
        alert('Inference error: ' + err.message);
      }
    }

    renderGenres();
    runInference();
  </script>
</body>
</html>
"""


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
