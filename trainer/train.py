"""
Audience Segmentation Trainer Service
Extracts behavioral features, encodes genres, scales numerical features,
runs KMeans experiments, fits the selected model, generates segment metadata,
and serializes artifacts for API serving.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add trainer directory to path to ensure proper module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import OTTFeaturePreprocessor, GENRE_LIST, NUMERICAL_FEATURES


def resolve_default_data_path() -> str:
    possible_paths = [
        os.environ.get("DATA_PATH", ""),
        "data/dataset.csv",
        "../data/dataset.csv",
        "/app/data/dataset.csv",
    ]
    for p in possible_paths:
        if p and os.path.exists(p):
            return p
    return "data/dataset.csv"


def resolve_default_model_dir() -> str:
    if "MODEL_DIR" in os.environ:
        return os.environ["MODEL_DIR"]
    if os.path.exists("/models"):
        return "/models"
    return "models"


def train_model(
    data_path: str = None,
    models_dir: str = None,
    random_state: int = 42,
    selected_k: int = 5,
):
    if data_path is None:
        data_path = resolve_default_data_path()
    if models_dir is None:
        models_dir = resolve_default_model_dir()

    print("=" * 60)
    print("STARTING AUDIENCE SEGMENTATION TRAINING PIPELINE")
    print(f"Data Path: {data_path}")
    print(f"Models Target Directory: {models_dir}")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    raw_df = pd.read_csv(data_path)
    initial_rows = len(raw_df)
    print(f"Loaded {initial_rows} rows from {data_path}")

    # 1. Data Cleaning & Validation (Deduplicate)
    clean_df = raw_df.drop_duplicates().copy()
    dedup_rows = len(clean_df)
    print(f"Removed {initial_rows - dedup_rows} duplicate rows. Remaining: {dedup_rows}")

    # 2. Fit Preprocessor
    preprocessor = OTTFeaturePreprocessor()
    preprocessor.fit(clean_df)
    X_processed = preprocessor.transform(clean_df)
    feature_names = preprocessor.feature_names_
    print(f"Processed Feature Matrix Shape: {X_processed.shape}")
    print(f"Features: {feature_names}")

    # 3. KMeans Cluster Count Experiments (K=2 to K=8)
    print("\nRunning KMeans Experiments across K in range [2, 8]...")
    experiments = []
    
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = km.fit_predict(X_processed)
        sil = float(silhouette_score(X_processed, cluster_labels))
        inertia = float(km.inertia_)
        counts = pd.Series(cluster_labels).value_counts().to_dict()
        
        experiments.append({
            "k": k,
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil, 4),
            "cluster_sizes": {int(c): int(cnt) for c, cnt in sorted(counts.items())}
        })
        print(f"  K={k} | Inertia={inertia:8.2f} | Silhouette={sil:.4f} | Sizes={dict(sorted(counts.items()))}")

    # Save experiments to CSV
    os.makedirs(models_dir, exist_ok=True)
    exp_df = pd.DataFrame(experiments)
    exp_csv_path = os.path.join(models_dir, "kmeans_experiments.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    # Also save in current working directory for root reference if different
    if models_dir != "." and not os.path.exists("kmeans_experiments.csv"):
        exp_df.to_csv("kmeans_experiments.csv", index=False)
    print(f"\nSaved KMeans experiment summary to {exp_csv_path}")

    # 4. Fit Final Selected KMeans Model
    print(f"\nFitting final production KMeans with selected K={selected_k}...")
    final_kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    clean_df["cluster_id"] = final_kmeans.fit_predict(X_processed)
    
    # 5. Analyze Cluster Profiles
    profiles = []
    segment_metadata = {}

    for c_id in range(selected_k):
        sub = clean_df[clean_df["cluster_id"] == c_id]
        size = len(sub)
        pct = (size / len(clean_df)) * 100.0
        
        # Calculate behavioral feature means
        mean_watch = float(sub["watch_time_hours"].mean())
        mean_session = float(sub["avg_session_mins"].mean())
        mean_sess_wk = float(sub["sessions_per_week"].mean())
        mean_comp = float(sub["completion_rate"].mean())
        mean_weekend = float(sub["weekend_usage_ratio"].mean())
        mean_div = float(sub["genre_diversity"].mean())
        
        # Calculate genre probabilities
        sub_genres = np.zeros(len(GENRE_LIST))
        for tg in sub["top_genres"]:
            if pd.isna(tg) or not str(tg).strip():
                continue
            toks = [t.strip() for t in str(tg).split("|") if t.strip()]
            for g_i, g in enumerate(GENRE_LIST):
                if g in toks:
                    sub_genres[g_i] += 1
        sub_genre_probs = sub_genres / size
        
        top_genre_indices = np.argsort(sub_genre_probs)[::-1][:3]
        top_genre_summary = [
            {"genre": GENRE_LIST[idx], "affinity_ratio": round(float(sub_genre_probs[idx]), 3)}
            for idx in top_genre_indices
        ]
        
        # Assign meaningful name and description based on behavioral statistics
        if mean_watch > 35.0 and mean_session > 80.0:
            seg_name = "Heavy Action & Sci-Fi Power Viewers"
            description = "High watch time, high frequency, long sessions; strong affinity for Action, Sci-Fi, and Thriller."
            strategy = "Prioritize high-octane blockbusters, franchise releases, and binge-ready action/sci-fi series."
            rec_genres = ["Action", "Sci-Fi", "Thriller"]
        elif mean_div > 5.0:
            seg_name = "High-Diversity Genre Explorers"
            description = "High genre diversity across catalog, frequent viewing sessions, wide curiosity."
            strategy = "Promote varied discovery, curated cross-genre bundles, and critically acclaimed releases."
            rec_genres = ["Romance", "Thriller", "Sci-Fi", "Documentary"]
        elif mean_weekend > 0.65 and mean_watch < 10.0:
            seg_name = "Casual Low-Activity Weekend Viewers"
            description = "Low total watch time, low weekly frequency, viewing concentrated heavily on weekends."
            strategy = "Surface popular trending titles, low-commitment comedies, and quick highlights."
            rec_genres = ["Comedy", "Drama"]
        elif mean_session < 35.0 and any(g["genre"] in ["Animation", "Family"] for g in top_genre_summary if g["affinity_ratio"] > 0.7):
            seg_name = "Short-Session Animation & Family Viewers"
            description = "Short average sessions, moderate weekly frequency, heavy consumption of Animation and Family content."
            strategy = "Recommend family-friendly features, animated shorts, and kid-friendly episodic content."
            rec_genres = ["Animation", "Family", "Comedy"]
        else:
            seg_name = "Focused Drama & Story Viewers"
            description = "High completion rate, deliberate long-form viewing, strong preference for Drama, Romance, and Thrillers."
            strategy = "Prioritize character-driven narratives, acclaimed drama miniseries, and romance films."
            rec_genres = ["Drama", "Romance", "Thriller"]

        profile_entry = {
            "cluster_id": c_id,
            "segment_name": seg_name,
            "cluster_size": size,
            "percentage_of_users": round(pct, 2),
            "mean_watch_time_hours": round(mean_watch, 2),
            "mean_avg_session_mins": round(mean_session, 2),
            "mean_sessions_per_week": round(mean_sess_wk, 2),
            "mean_completion_rate": round(mean_comp, 3),
            "mean_weekend_usage_ratio": round(mean_weekend, 3),
            "mean_genre_diversity": round(mean_div, 2),
            "top_genres": top_genre_summary,
            "description": description,
            "recommendation_strategy": strategy,
            "recommended_genres": rec_genres,
        }
        profiles.append(profile_entry)
        
        segment_metadata[str(c_id)] = {
            "segment_id": c_id,
            "segment_name": seg_name,
            "description": description,
            "recommendation_strategy": strategy,
            "recommended_genres": rec_genres,
            "cluster_size": size,
            "percentage_of_users": round(pct, 2),
            "centroid_stats": {
                "watch_time_hours": round(mean_watch, 2),
                "avg_session_mins": round(mean_session, 2),
                "sessions_per_week": round(mean_sess_wk, 2),
                "completion_rate": round(mean_comp, 3),
                "weekend_usage_ratio": round(mean_weekend, 3),
                "genre_diversity": round(mean_div, 2),
            }
        }

    # Save cluster profiles CSV
    profiles_df = pd.DataFrame(profiles)
    profiles_csv_path = os.path.join(models_dir, "cluster_profiles.csv")
    profiles_df.to_csv(profiles_csv_path, index=False)
    print(f"Saved cluster profiles to {profiles_csv_path}")

    # 6. Persist Pipeline and Artifacts
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    model_path = os.path.join(models_dir, "kmeans_model.joblib")
    meta_path = os.path.join(models_dir, "segment_metadata.json")
    results_path = os.path.join(models_dir, "training_results.json")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(final_kmeans, model_path)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(segment_metadata, f, indent=2)

    training_results = {
        "dataset_rows": initial_rows,
        "clean_rows": dedup_rows,
        "features": feature_names,
        "feature_count": len(feature_names),
        "selected_k": selected_k,
        "silhouette_score": round(float(silhouette_score(X_processed, clean_df["cluster_id"])), 4),
        "inertia": round(float(final_kmeans.inertia_), 2),
        "experiments": experiments,
        "cluster_profiles": profiles,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(training_results, f, indent=2)

    print(f"Successfully serialized preprocessor to: {preprocessor_path}")
    print(f"Successfully serialized KMeans model to: {model_path}")
    print(f"Successfully serialized segment metadata to: {meta_path}")
    print(f"Successfully serialized training results to: {results_path}")

    # 7. Test Reloading & Inference
    print("\nVerifying reload of persisted artifacts...")
    loaded_prep = joblib.load(preprocessor_path)
    loaded_km = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)

    test_sample = pd.DataFrame([{
        "user_id": "USR-8192",
        "watch_time_hours": 32.5,
        "avg_session_mins": 85.0,
        "sessions_per_week": 8,
        "completion_rate": 0.85,
        "weekend_usage_ratio": 0.35,
        "genre_diversity": 2,
        "top_genres": "Action|Thriller",
    }])

    test_X = loaded_prep.transform(test_sample)
    test_cluster = int(loaded_km.predict(test_X)[0])
    test_distances = loaded_km.transform(test_X)
    test_distance = float(test_distances[0][test_cluster])
    test_seg_name = loaded_meta[str(test_cluster)]["segment_name"]

    print(f"Inference Test on Sample USR-8192:")
    print(f"  Assigned Segment ID: {test_cluster}")
    print(f"  Segment Name: {test_seg_name}")
    print(f"  Distance to Centroid: {test_distance:.4f}")
    print("=" * 60)
    print("TRAINER EXECUTION AND VERIFICATION COMPLETE")
    print("=" * 60)

    return training_results


if __name__ == "__main__":
    train_model()
