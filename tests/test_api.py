"""
Comprehensive Automated Unit and Integration Tests for FastAPI Service.
Tests endpoints, schemas, validation rules, edge cases, and model determinism.
"""

import os
import sys
import copy
import pytest
from fastapi.testclient import TestClient

# Add project root and api dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from main import app, MODEL_STATE, load_artifacts


@pytest.fixture(scope="module")
def client():
    load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns 200 and model_loaded True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_recommend_valid_standard_request(client):
    """Test POST /recommend with valid sample payload from problem statement."""
    payload = {
        "user_id": "USR-8192",
        "watch_time_hours": 32.5,
        "top_genres": ["Action", "Thriller"],
        "avg_session_mins": 85.0,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "USR-8192"
    assert isinstance(data["segment_id"], int)
    assert 0 <= data["segment_id"] <= 4
    assert isinstance(data["segment_name"], str)
    assert len(data["segment_name"]) > 0
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
    assert isinstance(data["distance_to_centroid"], float)
    assert data["distance_to_centroid"] >= 0.0


def test_recommend_different_archetypes_yield_different_segments(client):
    """Verify different user profiles correctly map to different segments."""
    # Archetype 1: Heavy Action/Sci-Fi binge watcher
    p1 = {
        "user_id": "USR-HEAVY",
        "watch_time_hours": 50.0,
        "avg_session_mins": 100.0,
        "sessions_per_week": 12,
        "top_genres": ["Action", "Sci-Fi", "Thriller"],
    }
    # Archetype 2: Casual low-activity weekend viewer
    p2 = {
        "user_id": "USR-CASUAL",
        "watch_time_hours": 4.0,
        "avg_session_mins": 30.0,
        "sessions_per_week": 2,
        "weekend_usage_ratio": 0.8,
        "top_genres": ["Comedy"],
    }
    # Archetype 3: Short session animation family viewer
    p3 = {
        "user_id": "USR-FAMILY",
        "watch_time_hours": 10.0,
        "avg_session_mins": 25.0,
        "sessions_per_week": 4,
        "top_genres": ["Animation", "Family"],
    }

    r1 = client.post("/recommend", json=p1).json()
    r2 = client.post("/recommend", json=p2).json()
    r3 = client.post("/recommend", json=p3).json()

    assert r1["segment_id"] == 0  # Heavy Action & Sci-Fi Power Viewers
    assert r2["segment_id"] == 1  # Casual Low-Activity Weekend Viewers
    assert r3["segment_id"] == 4  # Short-Session Animation & Family Viewers
    assert r1["segment_id"] != r2["segment_id"]
    assert r2["segment_id"] != r3["segment_id"]


def test_recommend_empty_and_unknown_genres(client):
    """Test handling of empty top_genres and unseen genre values."""
    payload_empty = {
        "user_id": "USR-EMPTY-GENRE",
        "watch_time_hours": 20.0,
        "avg_session_mins": 45.0,
        "top_genres": [],
    }
    resp1 = client.post("/recommend", json=payload_empty)
    assert resp1.status_code == 200
    assert len(resp1.json()["recommendations"]) > 0

    payload_unknown = {
        "user_id": "USR-UNKNOWN-GENRE",
        "watch_time_hours": 20.0,
        "avg_session_mins": 45.0,
        "top_genres": ["K-Drama", "Anime-Super", "Cooking"],
    }
    resp2 = client.post("/recommend", json=payload_unknown)
    assert resp2.status_code == 200
    assert len(resp2.json()["recommendations"]) > 0


def test_recommend_negative_watch_time_rejected(client):
    """Test validation rejection for negative watch time."""
    payload = {
        "user_id": "USR-ERR",
        "watch_time_hours": -5.0,
        "avg_session_mins": 60.0,
        "top_genres": ["Action"],
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    assert "watch_time_hours" in response.json()["detail"]


def test_recommend_negative_session_mins_rejected(client):
    """Test validation rejection for negative session duration."""
    payload = {
        "user_id": "USR-ERR",
        "watch_time_hours": 10.0,
        "avg_session_mins": -10.0,
        "top_genres": ["Action"],
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    assert "avg_session_mins" in response.json()["detail"]


def test_recommend_empty_user_id_rejected(client):
    """Test validation rejection for empty user ID."""
    payload = {
        "user_id": "   ",
        "watch_time_hours": 10.0,
        "avg_session_mins": 30.0,
        "top_genres": ["Action"],
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_recommend_invalid_completion_rate_rejected(client):
    """Test validation rejection for out-of-bounds completion rate."""
    payload = {
        "user_id": "USR-ERR",
        "watch_time_hours": 10.0,
        "avg_session_mins": 30.0,
        "completion_rate": 1.5,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422


def test_recommend_malformed_json(client):
    """Test handling of malformed JSON payload."""
    response = client.post(
        "/recommend",
        content="This is not valid JSON",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_no_retraining_during_inference(client):
    """Verify that model weights / cluster centroids do NOT mutate during inference."""
    km = MODEL_STATE["kmeans_model"]
    initial_centers = copy.deepcopy(km.cluster_centers_)

    payload = {
        "user_id": "USR-REPEAT-TEST",
        "watch_time_hours": 32.5,
        "top_genres": ["Action", "Thriller"],
        "avg_session_mins": 85.0,
    }
    for _ in range(5):
        res = client.post("/recommend", json=payload)
        assert res.status_code == 200

    after_centers = MODEL_STATE["kmeans_model"].cluster_centers_
    import numpy as np
    assert np.array_equal(initial_centers, after_centers)
