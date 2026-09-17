# Containerized Audience Segmentation & Personalization Service

A lightweight, reproducible, CPU-friendly Machine Learning service that groups Over-The-Top (OTT) streaming users based on behavioral telemetry and delivers personalized content recommendations via a REST API.

---

## 🏗 Architecture Overview

```
                            [ Dataset: data/dataset.csv ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │            TRAINER SERVICE            │
                      │  • Deduplication & median imputation  │
                      │  • 15-Feature Engineering (6 num + 9) │
                      │  • StandardScaler normalization       │
                      │  • KMeans clustering (K=5 selected)   │
                      │  • Serializes pipeline to /models     │
                      └───────────────────┬───────────────────┘
                                          │
                                          │  (Shared Docker Volume: models_volume)
                                          ▼
                      ┌───────────────────────────────────────┐
                      │              API SERVICE              │
                      │  • FastAPI + Uvicorn                  │
                      │  • Loads persisted model artifacts    │
                      │  • Non-root user (appuser:1000)       │
                      │  • GET /health (Docker healthcheck)   │
                      │  • POST /recommend (no retraining)    │
                      └───────────────────┬───────────────────┘
                                          │
                                          │  (service_healthy condition)
                                          ▼
                      ┌───────────────────────────────────────┐
                      │           EVALUATOR SERVICE           │
                      │  • Polls /health until ready          │
                      │  • Tests 5 behavioral archetypes      │
                      │  • Tests 10 hostile & edge-case inputs│
                      │  • Verifies zero-drift determinism    │
                      │  • Generates metrics.json             │
                      └───────────────────────────────────────┘
```

---

## 🛠 Technology Stack

- **Language:** Python 3.11
- **Machine Learning:** Scikit-Learn 1.6.1 (`StandardScaler`, `KMeans`), NumPy 2.2.3, Pandas 2.2.3, Joblib 1.4.2
- **API Framework:** FastAPI 0.115.8, Uvicorn 0.34.0, Pydantic 2.10.6
- **Validation & Testing:** Pytest 8.3.4, HTTPX 0.28.1
- **Containerization:** Docker & Docker Compose v2

---

## 📁 Project Structure

```
Hackathon 3hours/
│
├── docker-compose.yml          # Orchestrates trainer, api, and evaluator services
├── README.md                   # Quickstart guide, API documentation, and instructions
├── REPORT.md                   # Detailed technical evaluation report
├── metrics.json                # Authoritative evaluation results generated automatically
├── kmeans_experiments.csv      # K=2 through K=8 inertia and silhouette score log
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git tracking exclusions
│
├── data/
│   └── dataset.csv             # OTT user activity telemetry dataset
│
├── trainer/
│   ├── Dockerfile              # Trainer container build definition
│   ├── requirements.txt        # Pinned trainer dependencies
│   ├── train.py                # Model training, K-selection, and artifact serializer
│   └── preprocessing.py        # Scikit-learn custom preprocessor transformer
│
├── api/
│   ├── Dockerfile              # FastAPI container build definition
│   ├── requirements.txt        # Pinned API dependencies
│   ├── main.py                 # FastAPI endpoints (/health, /recommend), validation
│   ├── preprocessing.py        # Preprocessing transformer for inference
│   └── recommendations.py      # Transparent rule-based recommendation engine
│
├── evaluator/
│   ├── Dockerfile              # Evaluator container build definition
│   ├── requirements.txt        # Pinned evaluator dependencies
│   └── evaluate.py             # Healthcheck poller, test suite, and metrics.json generator
│
├── models/                     # Shared model volume mount directory
│   ├── preprocessor.joblib     # Persisted fitted preprocessor
│   ├── kmeans_model.joblib     # Persisted fitted KMeans model (K=5)
│   ├── segment_metadata.json   # Discovered cluster profiles and strategies
│   ├── cluster_profiles.csv    # Cluster summaries table
│   └── training_results.json   # Training experiments and metrics log
│
├── samples/
│   ├── request.json            # Sample API request payload
│   └── response.json           # Sample API response payload
│
└── tests/
    └── test_api.py             # Automated unit and integration test suite
```

---

## 🚀 Quickstart & One-Command Startup

### Prerequisites
- Docker (version 20.10+) and Docker Compose installed and running.

### 1. Start the Full System
To run the trainer, boot the API, and execute the automated evaluator, run:

```bash
docker compose down -v && docker compose up --build
```

### 2. Startup Execution Sequence
1. `ott_trainer` starts, cleans the dataset, runs $K \in [2,8]$ experiments, fits the selected $K=5$ model, and writes artifacts to the shared volume `/models`. (Exits with code 0).
2. `ott_api` starts once `ott_trainer` finishes, loads the persisted model into memory, and becomes healthy on port `8000`.
3. `ott_evaluator` starts once `ott_api` is healthy, executes functional tests, validates edge cases, checks reproducibility, writes `metrics.json`, and exits with code 0.

---

## 📡 API Usage & Endpoints

### 1. Health Check (`GET /health`)
Verifies service availability and model loading status.

```bash
curl -X GET http://localhost:8000/health
```

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 2. Personalized Recommendation (`POST /recommend`)
Ingests user metrics and returns the assigned audience segment, distance to centroid, and tailored content titles.

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR-8192",
    "watch_time_hours": 32.5,
    "avg_session_mins": 85.0,
    "top_genres": ["Action", "Thriller"]
  }'
```

**Response (`200 OK`):**
```json
{
  "user_id": "USR-8192",
  "segment_id": 3,
  "segment_name": "Focused Drama & Story Viewers",
  "recommendations": [
    "Midnight Intrigue",
    "Silent Witness",
    "The Blind Spot",
    "Frenzy Point"
  ],
  "distance_to_centroid": 1.7198
}
```

---

## 🧪 Discovered Audience Segments ($K=5$)

| ID | Segment Name | Share (%) | Key Characteristics | Strategy |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Heavy Action & Sci-Fi Power Viewers** | 21.5% | High watch time (44.9h), 94.5m sessions, Action/Sci-Fi/Thriller | Prioritize blockbuster releases & high-octane franchises |
| **1** | **Casual Low-Activity Weekend Viewers** | 20.6% | Low watch time (5.7h), 71.9% weekend usage, Comedy/Drama | Surface trending comedies & quick highlights |
| **2** | **High-Diversity Genre Explorers** | 19.9% | Broad taste (6.4 genres), 30.1h watch time, Thriller/Romance/Sci-Fi/Doc | Recommend cross-genre discovery bundles |
| **3** | **Focused Drama & Story Viewers** | 15.4% | High completion (80.7%), long sessions (75.1m), Drama/Romance | Prioritize character-driven narratives & serialized dramas |
| **4** | **Short-Session Animation & Family Viewers** | 22.6% | Bite-sized sessions (27.3m), Animation/Family/Comedy | Recommend animated features & kid-friendly episodic content |

---

## 📊 Evaluation & `metrics.json`

The automated evaluator runs independent verification against the live containerized API and produces `metrics.json`.

**Summary of Measured Results:**
- **Clustering:** $K=5$, Silhouette Score = `0.3240`, Inertia = `2261.83`, Features = `15`, Samples = `1000`.
- **Cluster Balance Ratio:** `0.6814` (Min cluster = 154, Max cluster = 226).
- **API Correctness:** `5 / 5` representative profile archetypes passed.
- **Edge-Case Validation:** `10 / 10` boundary cases passed (including empty genres, unseen genres, negative values, type mismatches, and malformed JSON).
- **Inference Stability:** Zero drift across repeated requests.
- **Overall Status:** **`PASS`**.

---

## 🛑 Stopping the Services

To stop containers and clean up the shared network and volumes:

```bash
docker compose down -v
```

---

## ⚠️ Limitations

1. **Synthetic Telemetry:** Telemetry data represents a synthetic benchmark rather than live user logs.
2. **KMeans Geometry:** Assumes spherical cluster variances; complex non-linear structures could benefit from GMM or Density-Based Clustering in future iterations.
3. **Rule-Based Title Catalog:** Recommendations are mapped from segment-level genre priorities rather than a collaborative filtering matrix.
