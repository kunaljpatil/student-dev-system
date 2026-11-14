"""
Model Training and Serialization Script (spec Section 6.5, Phase 4).

Orchestrates:
1. Training dataset preparation (350 students Term 1 features and targets)
2. Train/test splitting (80/20 stratified)
3. Model 1: Student Performance Prediction Model (XGBoost Regressor + Ridge baseline + 5-fold CV)
4. Model 2: Student Performance Risk Classification Model (XGBoost Classifier + Logistic baseline)
5. Model 3: Student Development Profiling Model (K-Means + elbow/silhouette validation + centroid naming)
6. Feature Explainability (SHAP TreeExplainer & cohort global importance)
7. Artifact serialization to app/ml/artifacts/ using joblib
8. Metrics report generation at app/ml/artifacts/reports/metrics_report.json
9. Updating ml_model_info table in the database
"""

import os
import sys

# Ensure repository root is in python search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any

from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import MlModelInfo
from app.ml.features import FEATURE_NAMES, CLUSTER_FEATURE_NAMES
from app.ml.preprocessing import prepare_training_dataset, get_train_test_splits
from app.ml.train_regressor import train_regressor
from app.ml.train_classifier import train_classifier
from app.ml.train_clustering import train_clustering
from app.ml.explain import StudentExplainer


def run_training_pipeline(seed: int = 2024, model_version: str = "1.0.0") -> Dict[str, Any]:
    print("=" * 70)
    print("   AI-BASED STUDENT DEVELOPMENT SYSTEM - ML TRAINING PIPELINE")
    print(f"   Version: {model_version} | Seed: {seed}")
    print("=" * 70)

    # 1. Dataset preparation
    print("\n[*] Step 1/6: Preparing 350-student Term 1 feature vectors and targets...")
    X, y_reg, y_clf = prepare_training_dataset(seed=seed)
    print(f"    Dataset loaded: {len(X)} samples, {len(FEATURE_NAMES)} engineered features.")
    print(f"    Class balance: {y_clf.value_counts().to_dict()}")

    splits = get_train_test_splits(X, y_reg, y_clf, test_size=0.20, random_state=seed)
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_reg_train, y_reg_test = splits["y_reg_train"], splits["y_reg_test"]
    y_clf_train, y_clf_test = splits["y_clf_train"], splits["y_clf_test"]
    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)} (80/20 split)")

    # 2. Model 1: Regressor
    print("\n[*] Step 2/6: Training Model 1: Student Performance Prediction Model...")
    xgb_reg, ridge_base, reg_metrics = train_regressor(
        X_train, y_reg_train, X_test, y_reg_test, random_state=seed
    )
    print(f"    [XGBoost] R2: {reg_metrics['r2']} | MAE: {reg_metrics['mae']} | RMSE: {reg_metrics['rmse']}")
    print(f"    [5-Fold CV R2]: {reg_metrics['cv_r2_mean']} +/- {reg_metrics['cv_r2_std']}")
    print(f"    [Baseline Ridge] R2: {reg_metrics['baseline']['r2']} | MAE: {reg_metrics['baseline']['mae']}")

    # 3. Model 2: Classifier
    print("\n[*] Step 3/6: Training Model 2: Student Performance Risk Classification Model...")
    xgb_clf, log_base, clf_metrics = train_classifier(
        X_train, y_clf_train, X_test, y_clf_test, random_state=seed
    )
    print(f"    [XGBoost] Accuracy: {clf_metrics['accuracy']} | Macro F1: {clf_metrics['macro_f1']}")
    print(f"    [Baseline Logistic] Accuracy: {clf_metrics['baseline']['accuracy']} | Macro F1: {clf_metrics['baseline']['macro_f1']}")
    print(f"    [Confusion Matrix]: {clf_metrics['confusion_matrix']}")

    # 4. Model 3: Clustering
    print("\n[*] Step 4/6: Training Model 3: Student Development Profiling Model (K-Means)...")
    kmeans, cluster_scaler, cluster_metrics = train_clustering(
        X, k_range=(2, 6), fixed_k=3, random_state=seed
    )
    print(f"    [K-Means] Selected k: {cluster_metrics['selected_k']} | Silhouette: {cluster_metrics['silhouette_score']}")
    print("    [Cluster Profiles]:")
    for cid, cinfo in cluster_metrics["cluster_distribution"].items():
        print(f"      - Cluster {cid}: {cinfo['name']} ({cinfo['count']} students, {cinfo['pct']}%)")

    # 5. SHAP Explainer
    print("\n[*] Step 5/6: Initializing SHAP TreeExplainer & Global Importance...")
    explainer = StudentExplainer(xgb_reg, feature_names=FEATURE_NAMES)
    global_importance = explainer.compute_global_importance(X_test)
    print(f"    Top 3 Influential Global Features:")
    for f in global_importance[:3]:
        print(f"      - {f['label']} ({f['feature']}): importance {f['importance']}")

    # Test single-student explanation
    sample_exp = explainer.explain_student(X_test.iloc[0])
    print(f"    Sample student prediction explanation verified.")
    print(f"      Base Value: {sample_exp['base_value']}, Predicted: {sample_exp['predicted_score']}")
    print(f"      Top positive factor: {sample_exp['top_positive'][0]['label']} (+{sample_exp['top_positive'][0]['shap_value']})")
    if sample_exp['top_negative']:
        print(f"      Top negative factor: {sample_exp['top_negative'][0]['label']} ({sample_exp['top_negative'][0]['shap_value']})")

    # 6. Serialization & Report
    print("\n[*] Step 6/6: Serializing artifacts and generating reports...")
    artifacts_dir = os.path.join(settings.BASE_DIR, "app", "ml", "artifacts")
    reports_dir = os.path.join(artifacts_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Save models
    joblib.dump(xgb_reg, os.path.join(artifacts_dir, "regressor.joblib"))
    joblib.dump(ridge_base, os.path.join(artifacts_dir, "regressor_baseline.joblib"))
    joblib.dump(xgb_clf, os.path.join(artifacts_dir, "classifier.joblib"))
    joblib.dump(log_base, os.path.join(artifacts_dir, "classifier_baseline.joblib"))
    joblib.dump(kmeans, os.path.join(artifacts_dir, "clustering.joblib"))
    joblib.dump(cluster_scaler, os.path.join(artifacts_dir, "clustering_scaler.joblib"))

    # Consolidated metadata
    now_utc = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_version": model_version,
        "trained_at": now_utc,
        "random_seed": seed,
        "sample_count": len(X),
        "feature_names": FEATURE_NAMES,
        "cluster_feature_names": CLUSTER_FEATURE_NAMES,
        "cluster_names": cluster_metrics["cluster_names"],
        "regressor_metrics": reg_metrics,
        "classifier_metrics": clf_metrics,
        "clustering_metrics": cluster_metrics,
        "global_importance": global_importance,
    }
    joblib.dump(metadata, os.path.join(artifacts_dir, "metadata.joblib"))

    # Save metrics report JSON
    report_path = os.path.join(reports_dir, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved metrics report to: {report_path}")

    # Update Database ml_model_info
    print("    Updating database table 'ml_model_info'...")
    db = SessionLocal()
    try:
        model_info = db.query(MlModelInfo).filter(MlModelInfo.id == "singleton").first()
        if not model_info:
            model_info = MlModelInfo(id="singleton")
            db.add(model_info)

        model_info.version = model_version
        model_info.metrics_json = json.dumps({
            "regressor": reg_metrics,
            "classifier": clf_metrics,
            "clustering": cluster_metrics,
        })
        model_info.feature_importance_json = json.dumps(global_importance)
        model_info.cluster_names_json = json.dumps(cluster_metrics["cluster_names"])
        model_info.status = "ready"
        db.commit()
        print("    Database 'ml_model_info' updated successfully.")
    except Exception as e:
        db.rollback()
        print(f"    [WARN] Could not update ml_model_info in DB: {e}")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print("   ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
