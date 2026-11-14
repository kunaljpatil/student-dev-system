"""
Inference Pipeline & Prediction Service (spec Section 6.8).

Loads serialized model artifacts from app/ml/artifacts/ and provides:
1. predict_student: End-to-end multi-model inference + SHAP + recommendations + DB persistence
2. predict_cohort: Class-wide batch prediction + cohort aggregation
3. get_model_metadata: Current active model info and global metrics
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Student, MlPrediction, Recommendation
from app.ml.features import FEATURE_NAMES, CLUSTER_FEATURE_NAMES, build_features_from_db
from app.ml.preprocessing import INVERSE_LABEL_MAP
from app.ml.explain import StudentExplainer
from app.ml.recommend import generate_recommendations


class ModelService:
    """Singleton service to load and cache ML artifacts."""
    _instance: Optional["ModelService"] = None

    def __init__(self):
        artifacts_dir = os.path.join(settings.BASE_DIR, "app", "ml", "artifacts")
        self.regressor = joblib.load(os.path.join(artifacts_dir, "regressor.joblib"))
        self.classifier = joblib.load(os.path.join(artifacts_dir, "classifier.joblib"))
        self.clustering = joblib.load(os.path.join(artifacts_dir, "clustering.joblib"))
        self.clustering_scaler = joblib.load(os.path.join(artifacts_dir, "clustering_scaler.joblib"))
        self.metadata = joblib.load(os.path.join(artifacts_dir, "metadata.joblib"))
        self.explainer = StudentExplainer(self.regressor, feature_names=FEATURE_NAMES)
        self.model_version = self.metadata.get("model_version", "1.0.0")
        self.cluster_names = self.metadata.get("cluster_names", {})

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    @classmethod
    def reload(cls) -> "ModelService":
        cls._instance = ModelService()
        return cls._instance


def predict_student(
    student_id: str,
    term: str = "Term 1",
    db: Optional[Session] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Executes end-to-end inference flow for an individual student:
    Features -> Regressor -> Classifier -> Clustering -> SHAP -> Recommendations -> Save DB.
    """
    if db is None:
        raise ValueError("Database session is required for student feature retrieval.")

    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise ValueError(f"Student with ID '{student_id}' not found.")

    service = ModelService.get_instance()

    # 1. Feature extraction from DB
    feature_vector = build_features_from_db(student_id, term, db)
    df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
    feature_dict = {col: float(val) for col, val in zip(FEATURE_NAMES, feature_vector)}

    # 2. Model 1: Regressor (Predicted overall final score 0-100)
    raw_pred_score = float(service.regressor.predict(df)[0])
    predicted_score = round(float(np.clip(raw_pred_score, 0.0, 100.0)), 2)

    # 3. Model 2: Classifier (Risk category)
    class_idx = int(service.classifier.predict(df)[0])
    risk_category = INVERSE_LABEL_MAP.get(class_idx, "On Track")
    class_probs = service.classifier.predict_proba(df)[0]
    probabilities = {
        INVERSE_LABEL_MAP[i]: round(float(class_probs[i]), 3)
        for i in range(len(INVERSE_LABEL_MAP))
    }

    # 4. Model 3: Clustering (Development Profile)
    df_cluster = df[CLUSTER_FEATURE_NAMES].copy()
    scaled_cluster_features = service.clustering_scaler.transform(df_cluster)
    cluster_id = int(service.clustering.predict(scaled_cluster_features)[0])
    cluster_name = service.cluster_names.get(str(cluster_id), service.cluster_names.get(cluster_id, "Steady Progressors"))

    # 5. SHAP Explanations
    shap_res = service.explainer.explain_student(df)
    shap_json_str = json.dumps({
        "base_value": shap_res["base_value"],
        "predicted_score": predicted_score,
        "top_positive": shap_res["top_positive"],
        "top_negative": shap_res["top_negative"],
    })

    # 6. Recommendation Engine (max 5 supportive recommendations)
    recommendations_list = generate_recommendations(
        predicted_score=predicted_score,
        risk_category=risk_category,
        cluster_name=cluster_name,
        shap_explanation=shap_res,
        features=feature_dict,
    )

    # 7. Persistence to database
    prediction_record = None
    if persist:
        # Check existing prediction for (student_id, term)
        prediction_record = (
            db.query(MlPrediction)
            .filter(MlPrediction.student_id == student_id, MlPrediction.term == term)
            .first()
        )
        if not prediction_record:
            prediction_record = MlPrediction(
                student_id=student_id,
                term=term,
                predicted_score=predicted_score,
                risk_category=risk_category,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                shap_json=shap_json_str,
                model_version=service.model_version,
            )
            db.add(prediction_record)
            db.flush()
        else:
            prediction_record.predicted_score = predicted_score
            prediction_record.risk_category = risk_category
            prediction_record.cluster_id = cluster_id
            prediction_record.cluster_name = cluster_name
            prediction_record.shap_json = shap_json_str
            prediction_record.model_version = service.model_version
            prediction_record.generated_at = datetime.now(timezone.utc)
            # Remove old recommendations for this prediction
            db.query(Recommendation).filter(Recommendation.prediction_id == prediction_record.id).delete()
            db.flush()

        # Insert new recommendations
        for rec in recommendations_list:
            db_rec = Recommendation(
                student_id=student_id,
                prediction_id=prediction_record.id,
                category=rec["category"],
                text=rec["text"],
                priority=rec["priority"],
            )
            db.add(db_rec)

        db.commit()
        db.refresh(prediction_record)

    return {
        "student_id": student_id,
        "student_name": student.user.full_name if student.user else f"Student {student.roll_no}",
        "roll_number": str(student.roll_no),
        "class_id": student.class_id,
        "term": term,
        "predicted_score": predicted_score,
        "risk_category": risk_category,
        "probabilities": probabilities,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "shap": json.loads(shap_json_str),
        "recommendations": recommendations_list,
        "model_version": service.model_version,
        "feature_values": feature_dict,
        "prediction_id": prediction_record.id if prediction_record else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def predict_cohort(
    class_id: str,
    term: str = "Term 1",
    db: Optional[Session] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Executes batch inference across all students in a given class.
    Aggregates risk distribution, mean predicted score, and cluster counts.
    """
    if db is None:
        raise ValueError("Database session is required.")

    students = (
        db.query(Student)
        .filter(Student.class_id == class_id)
        .order_by(Student.roll_no.asc())
        .all()
    )
    if not students:
        raise ValueError(f"No students found for class ID '{class_id}'.")

    student_results = []
    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    cluster_counts: Dict[str, int] = {}
    predicted_scores = []

    for st in students:
        res = predict_student(st.id, term=term, db=db, persist=persist)
        student_results.append({
            "student_id": res["student_id"],
            "student_name": res["student_name"],
            "roll_number": str(st.roll_no),
            "predicted_score": res["predicted_score"],
            "risk_category": res["risk_category"],
            "cluster_name": res["cluster_name"],
            "top_positive": res["shap"]["top_positive"][:2],
            "top_negative": res["shap"]["top_negative"][:2],
            "recommendation_count": len(res["recommendations"]),
        })

        cat = res["risk_category"]
        risk_counts[cat] = risk_counts.get(cat, 0) + 1
        cname = res["cluster_name"]
        cluster_counts[cname] = cluster_counts.get(cname, 0) + 1
        predicted_scores.append(res["predicted_score"])

    total_count = len(students)
    avg_score = round(float(np.mean(predicted_scores)), 2) if predicted_scores else 0.0

    return {
        "class_id": class_id,
        "term": term,
        "total_students": total_count,
        "average_predicted_score": avg_score,
        "risk_distribution": {
            cat: {"count": count, "pct": round(count / total_count * 100, 1)}
            for cat, count in risk_counts.items()
        },
        "cluster_distribution": {
            cname: {"count": count, "pct": round(count / total_count * 100, 1)}
            for cname, count in cluster_counts.items()
        },
        "students": student_results,
    }
