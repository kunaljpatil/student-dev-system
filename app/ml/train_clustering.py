"""
Model 3: Student Development Profiling Model (spec Section 6.1, 6.5).

Algorithm: K-Means Clustering (unsupervised).
Purpose: Groups students into development profiles based on soft skills,
academic performance, and engagement.
Validation: Standard scaling, elbow method (inertia), silhouette scores across k in [2..6].
Profiles: Centroid-based naming matching spec Section 6.5.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from app.ml.features import CLUSTER_FEATURE_NAMES


def assign_cluster_names(
    centroids_df: pd.DataFrame,
    overall_mean: pd.Series,
) -> Dict[int, str]:
    """
    Assigns human-interpretable names to clusters based on their standardized
    and unstandardized centroid features relative to the overall cohort.
    
    Standard spec cluster names:
    - High Achiever - Well Rounded
    - Academically Strong - Low Engagement
    - Engaged - Needs Academic Support
    - At Risk - Low Engagement
    - Steady Progressors
    """
    cluster_names: Dict[int, str] = {}
    assigned_names: set = set()

    # Calculate relative differences from dataset mean
    diffs = centroids_df.sub(overall_mean, axis=1)

    for cluster_id in centroids_df.index:
        c_diff = diffs.loc[cluster_id]
        
        # Key dimension differentials
        acad_diff = (c_diff["avg_internal_score"] + c_diff["avg_midterm_score"]) / 2.0
        eng_diff = (c_diff["attendance_pct"] / 10.0 + c_diff["ontime_rate"] * 10.0 + c_diff["engagement_index"] / 10.0) / 3.0
        soft_diff = (
            c_diff["communication"] +
            c_diff["presentation"] +
            c_diff["confidence"] +
            c_diff["participation"] +
            c_diff["teamwork"]
        ) / 5.0

        if acad_diff > 4.0 and eng_diff > 1.0 and soft_diff > 0.2:
            candidate = "High Achiever - Well Rounded"
        elif acad_diff > 3.0 and (eng_diff < 0.0 or soft_diff < -0.1):
            candidate = "Academically Strong - Low Engagement"
        elif acad_diff < -3.0 and (eng_diff > 0.5 or soft_diff > 0.1):
            candidate = "Engaged - Needs Academic Support"
        elif acad_diff < -4.0 and eng_diff < -0.5:
            candidate = "At Risk - Low Engagement"
        else:
            candidate = "Steady Progressors"

        # Disambiguate if duplicate
        if candidate in assigned_names:
            if acad_diff >= 0 and "High Achiever - Well Rounded" not in assigned_names:
                candidate = "High Achiever - Well Rounded"
            elif acad_diff < 0 and "At Risk - Low Engagement" not in assigned_names:
                candidate = "At Risk - Low Engagement"
            elif "Steady Progressors" not in assigned_names:
                candidate = "Steady Progressors"
            else:
                candidate = f"{candidate} (Cohort {cluster_id + 1})"

        assigned_names.add(candidate)
        cluster_names[cluster_id] = candidate

    return cluster_names


def train_clustering(
    X: pd.DataFrame,
    k_range: Tuple[int, int] = (2, 6),
    fixed_k: int = 3,
    random_state: int = 2024,
) -> Tuple[KMeans, StandardScaler, Dict[str, Any]]:
    """
    Extracts cluster features, standardizes them, runs K-Means search over k_range,
    selects/fits the optimal k (or fixed_k), profiles centroids, and produces metrics.
    """
    # 1. Extract cluster features
    available_cols = [c for c in CLUSTER_FEATURE_NAMES if c in X.columns]
    X_cluster = X[available_cols].copy()

    # 2. Standard scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 3. Search over k_range for validation (Elbow & Silhouette)
    k_metrics = {}
    best_k = fixed_k
    best_sil = -1.0

    for k in range(k_range[0], k_range[1] + 1):
        km_test = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_test = km_test.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels_test))
        inertia = float(km_test.inertia_)
        k_metrics[k] = {
            "silhouette": round(sil, 3),
            "inertia": round(inertia, 2),
        }
        if sil > best_sil:
            best_sil = sil
            # If optimal silhouette is close to fixed_k, use optimal
            if 3 <= k <= 4:
                best_k = k

    # Ensure best_k is chosen reasonably (default 3 or 4)
    selected_k = fixed_k if fixed_k in k_metrics else best_k

    # 4. Fit final K-Means
    kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=15)
    final_labels = kmeans.fit_predict(X_scaled)
    final_sil = float(silhouette_score(X_scaled, final_labels))

    # 5. Centroid profiling in original feature scale
    centroids_orig = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=available_cols,
    )
    overall_mean = X_cluster.mean()
    cluster_names = assign_cluster_names(centroids_orig, overall_mean)

    # Cluster distribution
    counts = pd.Series(final_labels).value_counts().to_dict()
    cluster_distribution = {
        int(c_id): {
            "name": cluster_names[c_id],
            "count": int(counts.get(c_id, 0)),
            "pct": round(float(counts.get(c_id, 0) / len(X_cluster) * 100), 1),
        }
        for c_id in range(selected_k)
    }

    metrics = {
        "model_name": "Student Development Profiling Model (K-Means Clustering)",
        "selected_k": selected_k,
        "silhouette_score": round(final_sil, 3),
        "inertia": round(float(kmeans.inertia_), 2),
        "k_search_metrics": k_metrics,
        "cluster_distribution": cluster_distribution,
        "cluster_names": {int(k): v for k, v in cluster_names.items()},
        "feature_names": available_cols,
        "centroids": {
            int(c_id): {col: round(float(centroids_orig.loc[c_id, col]), 2) for col in available_cols}
            for c_id in range(selected_k)
        },
    }

    return kmeans, scaler, metrics
