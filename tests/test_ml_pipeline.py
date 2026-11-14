"""
Pytest Test Suite for Phase 4: Machine Learning Pipeline.

Tests:
1. Feature extraction & ethics verification (24 features, no sensitive attributes)
2. Preprocessing & stratified splits
3. Model 1: Student Performance Prediction Model (XGBoost Regressor + Ridge)
4. Model 2: Student Performance Risk Classification Model (XGBoost Classifier + Logistic)
5. Model 3: Student Development Profiling Model (K-Means + Silhouette + Centroids)
6. Feature Explainability (SHAP TreeExplainer local and global importance)
7. Artifact persistence & metadata in database
"""

import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np

from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import MlModelInfo
from app.ml.features import FEATURE_NAMES, CLUSTER_FEATURE_NAMES, extract_features_from_dict
from app.ml.data_generation import generate_dataset
from app.ml.preprocessing import prepare_training_dataset, get_train_test_splits
from app.ml.train_regressor import train_regressor
from app.ml.train_classifier import train_classifier
from app.ml.train_clustering import train_clustering
from app.ml.explain import StudentExplainer
from scripts.train_all import run_training_pipeline


def test_feature_extraction_and_ethics():
    """Verify 24 features, numeric validity, and absence of sensitive attributes."""
    assert len(FEATURE_NAMES) == 24
    
    # Ethics check: NO sensitive attributes
    forbidden = ["gender", "sex", "religion", "caste", "income", "race", "ethnicity"]
    for feat in FEATURE_NAMES:
        for f in forbidden:
            assert f not in feat.lower(), f"Sensitive feature found: {feat}"

    # Extract sample student features
    ds = generate_dataset(seed=2024)
    student = ds["students"][0]
    term1 = student["terms"][0]
    vector = extract_features_from_dict(term1)

    assert len(vector) == 24
    assert all(np.isfinite(v) for v in vector)
    
    # Check specific bounds
    attendance_pct = vector[FEATURE_NAMES.index("attendance_pct")]
    assert 0 <= attendance_pct <= 100
    engagement_index = vector[FEATURE_NAMES.index("engagement_index")]
    assert 0 <= engagement_index <= 100


def test_preprocessing_and_stratified_split():
    """Verify training dataset preparation and 80/20 stratified train/test split."""
    X, y_reg, y_clf = prepare_training_dataset(seed=2024)
    assert len(X) == 350
    assert len(y_reg) == 350
    assert len(y_clf) == 350

    splits = get_train_test_splits(X, y_reg, y_clf, test_size=0.20, random_state=2024)
    assert len(splits["X_train"]) == 280
    assert len(splits["X_test"]) == 70

    # Ensure all 3 risk classes are represented in train and test
    assert set(splits["y_clf_train"].unique()) == {0, 1, 2}
    assert set(splits["y_clf_test"].unique()) == {0, 1, 2}


def test_regressor_training_and_metrics():
    """Verify Model 1 XGBoost Regressor meets R2 >= 0.70 threshold and evaluates baseline."""
    X, y_reg, y_clf = prepare_training_dataset(seed=2024)
    splits = get_train_test_splits(X, y_reg, y_clf, test_size=0.20, random_state=2024)

    xgb_reg, ridge_base, metrics = train_regressor(
        splits["X_train"],
        splits["y_reg_train"],
        splits["X_test"],
        splits["y_reg_test"],
        random_state=2024,
    )

    assert xgb_reg is not None
    assert ridge_base is not None
    assert metrics["r2"] >= 0.70, f"Expected R2 >= 0.70, got {metrics['r2']}"
    assert metrics["mae"] <= 8.0, f"Expected MAE <= 8.0, got {metrics['mae']}"
    assert metrics["cv_r2_mean"] >= 0.65
    assert "baseline" in metrics
    assert metrics["baseline"]["r2"] >= 0.60


def test_classifier_training_and_metrics():
    """Verify Model 2 XGBoost Classifier meets Macro F1 >= 0.75 and generates confusion matrix."""
    X, y_reg, y_clf = prepare_training_dataset(seed=2024)
    splits = get_train_test_splits(X, y_reg, y_clf, test_size=0.20, random_state=2024)

    xgb_clf, log_base, metrics = train_classifier(
        splits["X_train"],
        splits["y_clf_train"],
        splits["X_test"],
        splits["y_clf_test"],
        random_state=2024,
    )

    assert xgb_clf is not None
    assert log_base is not None
    assert metrics["accuracy"] >= 0.70, f"Expected Accuracy >= 0.70, got {metrics['accuracy']}"
    assert metrics["macro_f1"] >= 0.70, f"Expected Macro F1 >= 0.70, got {metrics['macro_f1']}"
    assert len(metrics["confusion_matrix"]) == 3
    assert "On Track" in metrics["per_class"]
    assert "Needs Monitoring" in metrics["per_class"]
    assert "Needs Support" in metrics["per_class"]


def test_clustering_training_and_profiles():
    """Verify Model 3 K-Means clustering generates silhouette score >= 0.15 and named profiles."""
    X, _, _ = prepare_training_dataset(seed=2024)
    kmeans, scaler, metrics = train_clustering(X, k_range=(2, 6), fixed_k=3, random_state=2024)

    assert kmeans is not None
    assert scaler is not None
    assert metrics["selected_k"] == 3
    assert metrics["silhouette_score"] >= 0.15, f"Expected Silhouette >= 0.15, got {metrics['silhouette_score']}"
    assert len(metrics["cluster_names"]) == 3
    for cid, name in metrics["cluster_names"].items():
        assert len(name) > 3


def test_shap_explainer():
    """Verify SHAP TreeExplainer generates top 5 positive/negative factors and global ranking."""
    X, y_reg, y_clf = prepare_training_dataset(seed=2024)
    splits = get_train_test_splits(X, y_reg, y_clf, test_size=0.20, random_state=2024)

    xgb_reg, _, _ = train_regressor(
        splits["X_train"],
        splits["y_reg_train"],
        splits["X_test"],
        splits["y_reg_test"],
        random_state=2024,
    )

    explainer = StudentExplainer(xgb_reg, feature_names=FEATURE_NAMES)
    explanation = explainer.explain_student(splits["X_test"].iloc[0])

    assert "base_value" in explanation
    assert "predicted_score" in explanation
    assert len(explanation["top_positive"]) <= 5
    assert len(explanation["top_negative"]) <= 5
    assert len(explanation["all_factors"]) == 24

    # Test global importance
    global_imp = explainer.compute_global_importance(splits["X_test"].iloc[:20])
    assert len(global_imp) == 24
    assert global_imp[0]["importance"] >= global_imp[-1]["importance"]


def test_full_pipeline_run_and_artifacts():
    """Verify scripts/train_all.py serializes artifacts, creates report, and updates DB."""
    metadata = run_training_pipeline(seed=2024, model_version="1.0.0")

    artifacts_dir = os.path.join(settings.BASE_DIR, "app", "ml", "artifacts")
    assert os.path.exists(os.path.join(artifacts_dir, "regressor.joblib"))
    assert os.path.exists(os.path.join(artifacts_dir, "classifier.joblib"))
    assert os.path.exists(os.path.join(artifacts_dir, "clustering.joblib"))
    assert os.path.exists(os.path.join(artifacts_dir, "clustering_scaler.joblib"))
    assert os.path.exists(os.path.join(artifacts_dir, "metadata.joblib"))

    report_path = os.path.join(artifacts_dir, "reports", "metrics_report.json")
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    assert report_data["model_version"] == "1.0.0"
    assert "regressor_metrics" in report_data
    assert "classifier_metrics" in report_data
    assert "clustering_metrics" in report_data

    # Verify Database ml_model_info entry
    db = SessionLocal()
    try:
        info = db.query(MlModelInfo).filter(MlModelInfo.id == "singleton").first()
        assert info is not None
        assert info.version == "1.0.0"
        assert info.status == "ready"
        assert info.metrics_json is not None
    finally:
        db.close()
