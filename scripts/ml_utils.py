"""
ml_utils.py - Reusable utilities and features for FlyRank content decay research.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

# Feature taxonomies
MODEL_NUMERIC_FEATURES = [
    'content_age_days', 'days_since_last_update', 'word_count', 'search_volume',
    'cpc', 'competition', 'avg_position', 'ctr', 'engagement_rate',
    'scroll_rate', 'ai_traffic_pct', 'days_with_impressions', 'days_with_sessions'
]

MODEL_LOG_FEATURES = [
    'impressions_90d', 'clicks_90d', 'pageviews_90d', 'sessions_90d', 'users_90d'
]

MODEL_CATEGORICAL_FEATURES = [
    'content_type', 'main_intent', 'position_tier', 'age_tier', 'impression_tier'
]

# Explicitly excluded to prevent target leakage
PROHIBITED_LEAKAGE_COLUMNS = [
    'trend_pct', 'trend_direction',
    'impressions_last_30d', 'clicks_last_30d', 'sessions_last_30d',
    'impressions_prev_30d', 'clicks_prev_30d', 'sessions_prev_30d',
    'provider_used', 'model_used'
]


def resolve_data_path(relative_path="data/raw/content_refresh_anonymized.csv"):
    """
    Robustly locate data file whether run from repo root or notebooks/scripts.
    """
    candidates = [
        relative_path,
        os.path.join("..", relative_path),
        os.path.join("..", "..", relative_path),
        os.path.join(os.path.dirname(__file__), "..", relative_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Could not locate dataset at {relative_path} from {os.getcwd()}")


def precision_at_k(scores, labels, k):
    """
    Compute Precision@K on ranking scores.
    """
    order = np.argsort(-np.asarray(scores))
    return float(np.asarray(labels)[order[:k]].mean())


def build_feature_matrix(df):
    """
    Build leak-free feature matrix X and label vector y.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Numerics and missingness flags
    for col in MODEL_NUMERIC_FEATURES:
        X[col] = df[col].fillna(0).astype(float)
        X[f'has_{col}'] = df[col].notnull().astype(float)

    # 2. Log1p compression on heavy-tailed totals
    for col in MODEL_LOG_FEATURES:
        X[f'log_{col}'] = np.log1p(df[col].fillna(0).astype(float))

    # 3. Categoricals
    for col in MODEL_CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col].fillna('unknown'), prefix=col, drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    # Ground truth binary label
    y = (df['trend_direction'] == 'down').astype(int)

    # Assert zero target leakage
    leak_check = set(X.columns) & set(PROHIBITED_LEAKAGE_COLUMNS)
    assert len(leak_check) == 0, f"Fatal leakage detected in features: {leak_check}"

    return X, y
