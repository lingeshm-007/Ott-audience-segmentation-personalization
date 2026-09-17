"""
Shared Preprocessing Module for OTT User Activity Dataset.
Contains feature names, valid genres, and the Scikit-Learn transformer.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


GENRE_LIST: List[str] = [
    "Action",
    "Animation",
    "Comedy",
    "Documentary",
    "Drama",
    "Family",
    "Romance",
    "Sci-Fi",
    "Thriller",
]

NUMERICAL_FEATURES: List[str] = [
    "watch_time_hours",
    "avg_session_mins",
    "sessions_per_week",
    "completion_rate",
    "weekend_usage_ratio",
    "genre_diversity",
]


class OTTFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for OTT User Activity data.
    Handles missing value imputation, multi-hot genre extraction,
    and standard scaling of numerical features.
    """

    def __init__(self, median_completion_rate: float = 0.6980):
        self.median_completion_rate = median_completion_rate
        self.scaler = StandardScaler()
        self.genres = GENRE_LIST
        self.numerical_cols = NUMERICAL_FEATURES
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if "completion_rate" in X.columns:
            median_val = X["completion_rate"].dropna().median()
            if not pd.isna(median_val):
                self.median_completion_rate = float(median_val)
        
        X_num = X[self.numerical_cols].copy()
        X_num["completion_rate"] = X_num["completion_rate"].fillna(self.median_completion_rate)
        
        self.scaler.fit(X_num)
        
        genre_cols = [f"genre_{g.lower().replace('-', '_')}" for g in self.genres]
        self.feature_names_ = list(self.numerical_cols) + genre_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_num = X[self.numerical_cols].copy()
        X_num["completion_rate"] = X_num["completion_rate"].fillna(self.median_completion_rate)
        
        # Clip numerical values to sensible valid bounds
        X_num["watch_time_hours"] = X_num["watch_time_hours"].clip(lower=0.0)
        X_num["avg_session_mins"] = X_num["avg_session_mins"].clip(lower=0.0)
        X_num["sessions_per_week"] = X_num["sessions_per_week"].clip(lower=0.0)
        X_num["completion_rate"] = X_num["completion_rate"].clip(lower=0.0, upper=1.0)
        X_num["weekend_usage_ratio"] = X_num["weekend_usage_ratio"].clip(lower=0.0, upper=1.0)
        X_num["genre_diversity"] = X_num["genre_diversity"].clip(lower=0.0, upper=float(len(self.genres)))
        
        scaled_num = self.scaler.transform(X_num)
        
        # Handle top_genres multi-hot encoding
        n_samples = len(X)
        genre_matrix = np.zeros((n_samples, len(self.genres)), dtype=float)
        
        for i, val in enumerate(X["top_genres"]):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if isinstance(val, list):
                tokens = [str(g).strip() for g in val if str(g).strip()]
            else:
                tokens = [str(g).strip() for g in str(val).split("|") if str(g).strip()]
            for g_idx, genre in enumerate(self.genres):
                if genre in tokens:
                    genre_matrix[i, g_idx] = 1.0
                    
        return np.hstack([scaled_num, genre_matrix])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
