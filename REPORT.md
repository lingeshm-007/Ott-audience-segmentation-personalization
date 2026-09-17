# OTT Audience Segmentation & Personalization Service — Technical Report

---

## 1. Problem Understanding

Modern Over-The-Top (OTT) streaming platforms generate continuous behavioral telemetry covering watch time, session duration, viewing frequency, completion rates, weekend habits, and genre affinities. To improve user engagement and retention without invasive tracking, platforms require automated, unsupervised mechanisms to group users exhibiting similar consumption patterns and deliver tailored content recommendations.

This project implements a lightweight, reproducible, CPU-friendly machine learning service that:
1. Cleans and transforms raw OTT activity telemetry into normalized behavioral and genre affinity features.
2. Discovers natural audience clusters using unsupervised $K$-Means clustering without artificial target labels.
3. Automatically interprets cluster centroids to assign meaningful, human-readable segment profiles.
4. Serves segmentation and transparent, rule-based recommendations through a high-performance FastAPI service without retraining during inference.
5. Employs an independent evaluator to validate API reliability, edge cases, and inference reproducibility within a containerized Docker Compose architecture.

---

## 2. Assumptions

1. **Telemetry Frequency:** Telemetry metrics (such as `watch_time_hours` and `sessions_per_week`) represent aggregate monthly user behavior.
2. **Feature Non-Negativity:** Viewing metrics (time, counts, durations) cannot be negative; ratios (completion, weekend viewing) reside strictly in the range $[0.0, 1.0]$.
3. **Genre Space Completeness:** The platform's active catalog consists of 9 core genres (`Action`, `Animation`, `Comedy`, `Documentary`, `Drama`, `Family`, `Romance`, `Sci-Fi`, `Thriller`). Unseen genres are handled gracefully without breaking inference.
4. **Transparent Personalization:** Recommendations are generated via a deterministic, explainable rule engine that maps audience segments and user affinities to catalog titles, avoiding black-box LLM hallucinations.
5. **No Cold-Start Retraining:** Model artifacts are serialized at build/training time; the inference API acts purely as a stateless scoring engine.

---

## 3. Dataset & Preprocessing

The raw dataset was loaded from `data/dataset.csv`:
- **Original Dataset Dimensions:** 1,005 rows $\times$ 8 columns.
- **Identifier Exclusion:** `user_id` was strictly excluded from clustering to prevent the model from learning arbitrary user hashes.
- **Deduplication:** Exactly 5 duplicate records (`USR-00412`, `USR-00522`, `USR-00661`, `USR-00738`, `USR-00741`) were identified and removed, yielding 1,000 clean user records.
- **Missing Value Imputation:** 
  - `completion_rate` was missing in 6 records (0.6%). It was imputed using the training set median (`0.6980`) to prevent distribution skewing.
  - `top_genres` was empty in 6 records (0.6%). These were encoded as zero-vectors across all genre flags without dropping the user records.
- **Numerical Validation & Bounding:** All continuous variables were verified non-negative and clipped to valid ranges during preprocessing.

---

## 4. Feature Engineering

A total of 15 features were engineered across two primary categories:

### A. Behavioral Continuous Features (6 Features)
1. `watch_time_hours`: Total monthly watch duration. Distinguishes power binge-watchers from casual drop-in viewers.
2. `avg_session_mins`: Average length of an individual viewing session. Distinguishes long-form cinematic viewers from short-form episodic viewers.
3. `sessions_per_week`: Frequency of platform visits per week. Measures habituation and platform reliance.
4. `completion_rate`: Ratio of started content completed to finish. Measures attention span and content commitment.
5. `weekend_usage_ratio`: Fraction of viewing that occurs on weekends vs. weekdays. Separates weekend leisure viewers from regular daily consumers.
6. `genre_diversity`: Count of distinct genres explored. Measures content curiosity versus narrow topical focus.

### B. Multi-Hot Genre Affinity Features (9 Features)
Binary indicator features for each discovered genre:
- `genre_action`, `genre_animation`, `genre_comedy`, `genre_documentary`, `genre_drama`, `genre_family`, `genre_romance`, `genre_sci_fi`, `genre_thriller`.

**Final Transformed Matrix Dimension:** `(1000, 15)`.

---

## 5. Scaling

Because $K$-Means relies on Euclidean distance calculations:
$$d(x, c) = \sqrt{\sum_{i=1}^{D} (x_i - c_i)^2}$$
features with large natural ranges (e.g., `avg_session_mins` $\approx 15-120$) would dominate features with smaller scales (e.g., `completion_rate` $\approx 0.3-1.0$). 

To ensure equal weighting across all behavioral dimensions, `StandardScaler` was fitted on the numerical columns:
$$z = \frac{x - \mu}{\sigma}$$
The fitted `StandardScaler` was integrated directly into a custom Scikit-Learn transformer (`OTTFeaturePreprocessor`) and persisted for production inference.

---

## 6. Clustering Model

- **Algorithm:** $K$-Means Clustering (`sklearn.cluster.KMeans`).
- **Initialization:** `k-means++` with `n_init=10`.
- **Random Seed:** `random_state=42` for complete determinism and reproducibility across environments.
- **Max Iterations:** `300`.
- **Convergence Tolerance:** `1e-4`.

---

## 7. $K$ Selection & Evidence

Experiments were conducted across $K \in [2, 8]$ using fixed seed `random_state=42`:

| $K$ | Inertia | Silhouette Score | Cluster Sizes |
| :---: | :---: | :---: | :--- |
| **$K=2$** | 4428.97 | **0.3858** | `{0: 565, 1: 435}` |
| **$K=3$** | 3296.51 | 0.3831 | `{0: 369, 1: 432, 2: 199}` |
| **$K=4$** | 2733.99 | 0.3658 | `{0: 431, 1: 199, 2: 155, 3: 215}` |
| **$K=5$ (Selected)** | **2261.83** | **0.3240** | **`{0: 215, 1: 206, 2: 199, 3: 154, 4: 226}`** |
| **$K=6$** | 2160.96 | 0.2870 | `{0: 199, 1: 154, 2: 118, 3: 215, 4: 120, 5: 194}` |
| **$K=7$** | 2075.42 | 0.2731 | `{0: 199, 1: 110, 2: 154, 3: 215, 4: 104, 5: 100, 6: 118}` |
| **$K=8$** | 2005.17 | 0.2142 | `{0: 118, 1: 109, 2: 102, 3: 199, 4: 153, 5: 110, 6: 102, 7: 107}` |

### Selection Rationale ($K=5$)
1. **Silhouette Reality:** While $K=2$ achieves the mathematically highest silhouette score (`0.3858`), two clusters only distinguish "high vs. low" activity, failing to capture distinct genre preferences or session formats.
2. **Elbow Behavior:** Inertia drops steeply from $K=2 \to 5$ ($4428.97 \to 2261.83$, a ~49% reduction). Beyond $K=5$, the elbow curve flattens significantly ($K=5 \to 6$ yields a marginal decrease of only 100.8).
3. **Cluster Size Balance:** In $K=4$, a single oversized cluster comprises 43.1% of the dataset (431 users), conflating short-session family viewers with casual weekend comedy viewers. In $K=5$, the clusters distribute exceptionally evenly across the user population (ranging between 15.4% and 22.6%, with a min/max balance ratio of **0.6814**).
4. **Actionable Business Segmentation:** $K=5$ cleanly isolates five distinct, interpretable OTT viewer personas that map directly to operational marketing and content strategies.

---

## 8. Cluster Profiles & Segment Interpretations

### Segment 0: Heavy Action & Sci-Fi Power Viewers
- **Cluster Size:** 215 users (21.5%)
- **Behavioral Characteristics:** Watch Time = 44.9 hrs/mo, Avg Session = 94.5 mins, Sessions/Wk = 10.5, Completion Rate = 83.5%, Weekend Ratio = 35.1%, Diversity = 3.0.
- **Top Genre Affinities:** Sci-Fi (90.2%), Action (88.8%), Thriller (87.0%).
- **Reasoning:** Highest engagement, frequency, and session length across the platform, with near-universal affinity for high-adrenaline genres.
- **Strategy:** Prioritize blockbuster action franchises, sci-fi series releases, and binge-ready thriller seasons.

### Segment 1: Casual Low-Activity Weekend Viewers
- **Cluster Size:** 206 users (20.6%)
- **Behavioral Characteristics:** Watch Time = 5.7 hrs/mo, Avg Session = 37.4 mins, Sessions/Wk = 2.1, Completion Rate = 49.8%, Weekend Ratio = 71.9%, Diversity = 1.96.
- **Top Genre Affinities:** Comedy (69.4%), Drama (66.0%), Family (58.3%).
- **Reasoning:** Lowest weekly engagement and completion rate, with over 71% of all viewing concentrated exclusively on weekends.
- **Strategy:** Surface low-friction comedies, bite-sized entertainment, and trending weekend highlights.

### Segment 2: High-Diversity Genre Explorers
- **Cluster Size:** 199 users (19.9%)
- **Behavioral Characteristics:** Watch Time = 30.1 hrs/mo, Avg Session = 62.1 mins, Sessions/Wk = 7.5, Completion Rate = 67.6%, Weekend Ratio = 39.5%, Diversity = 6.40.
- **Top Genre Affinities:** Thriller (74.4%), Romance (74.4%), Sci-Fi (72.4%), Documentary (50.3%).
- **Reasoning:** Defined by catalog curiosity (mean diversity of 6.4 genres per user vs. ~2.0 across other clusters) and consistent viewing habits.
- **Strategy:** Surface cross-genre discovery bundles, hidden gems, and critically acclaimed releases.

### Segment 3: Focused Drama & Story Viewers
- **Cluster Size:** 154 users (15.4%)
- **Behavioral Characteristics:** Watch Time = 26.5 hrs/mo, Avg Session = 75.1 mins, Sessions/Wk = 6.5, Completion Rate = 80.7%, Weekend Ratio = 40.2%, Diversity = 1.94.
- **Top Genre Affinities:** Romance (65.6%), Thriller (63.0%), Drama (59.1%).
- **Reasoning:** Long deliberate sessions (75.1 mins) and high completion rate (80.7%) focused tightly on narrative-rich Drama, Romance, and Suspense.
- **Strategy:** Recommend character-driven dramas, romantic features, and serialized story arcs.

### Segment 4: Short-Session Animation & Family Viewers
- **Cluster Size:** 226 users (22.6%)
- **Behavioral Characteristics:** Watch Time = 11.4 hrs/mo, Avg Session = 27.3 mins, Sessions/Wk = 4.1, Completion Rate = 63.1%, Weekend Ratio = 52.7%, Diversity = 3.62.
- **Top Genre Affinities:** Animation (85.8%), Comedy (85.0%), Family (81.9%).
- **Reasoning:** Characterized by short bite-sized sessions (27.3 mins) and strong dominance in Animation, Family, and Comedy titles (typical household/children viewing patterns).
- **Strategy:** Recommend animated shorts, family movie nights, and kid-friendly comedy series.

---

## 9. API Design & Implementation

Built with FastAPI and Pydantic:
- **`GET /health`**: Returns `{"status": "ok", "model_loaded": true}` after checking loaded artifacts.
- **`POST /recommend`**: Ingests user activity metrics, executes preprocessor scaling and multi-hot extraction, computes $K$-Means cluster assignment and distance to centroid, and returns recommendations.

### Request Schema
```json
{
  "user_id": "USR-8192",
  "watch_time_hours": 32.5,
  "avg_session_mins": 85.0,
  "top_genres": ["Action", "Thriller"],
  "sessions_per_week": 8.0,
  "completion_rate": 0.85,
  "weekend_usage_ratio": 0.35,
  "genre_diversity": 2.0
}
```

### Response Schema
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

### Distance to Centroid
Calculated directly from the $K$-Means model in the transformed 15-dimensional standardized feature space:
$$\text{distance} = \| X_{\text{user}} - C_{\text{predicted}} \|_2$$

### Recommendation Logic
A transparent rule-based matching engine prioritizes:
1. User's specific preferred genres intersecting with the predicted segment's target genres.
2. Primary recommended genres from the cluster metadata profile.
3. Fallback popular titles if genre inputs are empty or unrecognized.

### No Retraining Contract
Persisted artifacts (`preprocessor.joblib`, `kmeans_model.joblib`, `segment_metadata.json`) are loaded strictly once during application startup. Model centroids were verified to remain immutable during inference.

---

## 10. Docker Architecture

The service is orchestrated as a three-container pipeline via `docker-compose.yml`:

```
┌────────────────────────────────────────────────────────┐
│                        trainer                         │
│  • Reads data/dataset.csv                              │
│  • Cleans, preprocesses, runs KMeans experiments       │
│  • Writes persisted model artifacts to /models         │
└──────────────────────────┬─────────────────────────────┘
                           │ (Shared Docker Volume: models_volume)
                           ▼
┌────────────────────────────────────────────────────────┐
│                          api                           │
│  • depends_on: trainer (service_completed_success)     │
│  • Loads /models artifacts into memory                 │
│  • Serves GET /health and POST /recommend on :8000     │
│  • Runs as non-root user (appuser:1000)                │
└──────────────────────────┬─────────────────────────────┘
                           │ (Docker Healthcheck: GET /health)
                           ▼
┌────────────────────────────────────────────────────────┐
│                       evaluator                        │
│  • depends_on: api (service_healthy)                   │
│  • Polls /health and verifies readiness                │
│  • Tests 5 profiles, 10 edge cases, 5 repro runs       │
│  • Automatically generates metrics.json                │
└────────────────────────────────────────────────────────┘
```

- **Base Image:** `python:3.11-slim`.
- **Pinned Dependencies:** Explicit versions for all packages (`numpy==2.2.3`, `pandas==2.2.3`, `scikit-learn==1.6.1`, `joblib==1.4.2`, `fastapi==0.115.8`, `uvicorn==0.34.0`, `pydantic==2.10.6`, `httpx==0.28.1`, `pytest==8.3.4`).
- **Security:** Non-root execution (`appuser`, UID `1000`).

---

## 11. Evaluator Methodology

The independent evaluator service:
1. Polls `GET /health` with timeout and backoff to ensure readiness without relying on arbitrary sleep commands.
2. Submits synthetic profiles corresponding to all 5 discovered segments and asserts expected outputs.
3. Tests 10 hostile and edge-case payloads to confirm robust HTTP status codes (`200` vs. `422`).
4. Executes repeated inference iterations to prove zero-drift determinism.
5. Ingests model training statistics and writes the authoritative `metrics.json`.

---

## 12. Evaluation Results (From Live `metrics.json`)

All metrics below are derived directly from the automated evaluation suite:

- **Selected $K$:** `5`
- **Silhouette Score:** `0.3240`
- **Inertia:** `2261.83`
- **Feature Dimension:** `15`
- **Evaluated Dataset Samples:** `1,000` (deduplicated)
- **Cluster Sizes:** `{0: 215, 1: 206, 2: 199, 3: 154, 4: 226}`
- **Cluster Proportions:** `{0: 21.5%, 1: 20.6%, 2: 19.9%, 3: 15.4%, 4: 22.6%}`
- **Cluster Balance Ratio ($\text{Min} / \text{Max}$):** **`0.6814`** ($154 / 226$)
- **Functional Profile Tests:** **`5 / 5 PASSED`**
- **Edge-Case Tests:** **`10 / 10 PASSED`**
- **Reproducibility Test:** **`PASS`** (Zero drift across 5 iterations)
- **Evaluator Exit Status:** `0` (Success)
- **Overall System Status:** **`PASS`**

---

## 13. Edge Cases Tested

| Test Case Name | Input Conditions | Expected Status | Actual Status | Result |
| :--- | :--- | :---: | :---: | :---: |
| `empty_top_genres` | Empty list `top_genres: []` | 200 OK | 200 OK | **PASS** |
| `unknown_unseen_genre`| Unrecognized tokens `["K-Drama", "Esports"]` | 200 OK | 200 OK | **PASS** |
| `zero_watch_time` | Zero metrics (`watch_time: 0.0`, `session: 0.0`) | 200 OK | 200 OK | **PASS** |
| `large_valid_values` | Large continuous values (`watch_time: 200.0`) | 200 OK | 200 OK | **PASS** |
| `missing_required_user_id` | Payload without `user_id` | 422 Error | 422 Error | **PASS** |
| `negative_watch_time` | Negative duration (`watch_time: -12.5`) | 422 Error | 422 Error | **PASS** |
| `negative_session_mins` | Negative duration (`avg_session: -45.0`) | 422 Error | 422 Error | **PASS** |
| `invalid_numeric_type`| String for numeric field (`"ten_hours"`) | 422 Error | 422 Error | **PASS** |
| `invalid_completion_rate` | Out-of-bounds ratio (`completion_rate: 1.75`) | 422 Error | 422 Error | **PASS** |
| `malformed_json_body` | Invalid JSON syntax payload | 422 Error | 422 Error | **PASS** |

---

## 14. Limitations

1. **Synthetic Telemetry Scope:** The dataset contains 1,005 synthetic user activity records. Real-world telemetry exhibits seasonal volatility, episodic content launches, and streaming quality metrics not represented here.
2. **KMeans Spherical Geometry:** $K$-Means assumes isotropic, spherical cluster variances. While feature scaling mitigates scale discrepancies, non-linear cluster boundaries may be better captured by Gaussian Mixture Models (GMM) or hierarchical clustering in future iterations.
3. **Static Recommendation Catalog:** The content catalog is currently rule-based and catalog-bounded. Integrating collaborative filtering or matrix factorization as a downstream ranking stage would enhance title variety.
4. **Offline Evaluation Boundary:** The evaluation suite tests synthetic profiles and edge cases, but does not simulate live streaming traffic concurrency at massive scale.

---

## 15. Reproducibility Instructions

The entire system is reproducible with a single command from the project root:

```bash
docker compose down -v && docker compose up --build
```

---

## 16. Development Progression & Decisions (What We Tried and Changed)

- **Phase 1 (Inspection):** Discovered 5 exact duplicate records and 6 missing `completion_rate` values. Identified 9 discrete genres across pipe-delimited strings.
- **Phase 2 (Trainer & Feature Design):** Tested $K \in [2, 8]$. Evaluated trade-offs between $K=2$ (highest silhouette, but coarse) and $K=5$ (balanced, interpretable, elbow inflection). Selected $K=5$ and persisted artifacts.
- **Phase 3 (FastAPI):** Built clean Pydantic schemas, custom exception handlers to prevent stack trace leaks, and confirmed zero model retraining during inference.
- **Phase 4 (Docker Orchestration):** Designed a 3-tier Compose setup with shared read-only volume mounts and native healthchecks.
- **Phase 5 (Evaluator & Metrics):** Implemented health polling, 5-archetype functional tests, 10 boundary tests, and automated generation of `metrics.json`.
