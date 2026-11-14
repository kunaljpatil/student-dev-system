"""
ML API Router (spec Section 7).

Endpoints:
- POST /api/ml/predict/{student_id} : On-demand prediction for a student
- POST /api/ml/predict-cohort/{class_id} : Cohort batch prediction for a class
- GET /api/ml/model-info : Model metadata, version, metrics, and global SHAP importance
"""

import json
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, Class, MlModelInfo
from app.core.dependencies import (
    get_current_user,
    require_staff,
    verify_student_access,
)
from app.schemas.ml import (
    PredictionResponse,
    CohortPredictionResponse,
    ModelInfoResponse,
)
from app.ml.predict import predict_student, predict_cohort, ModelService

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.post("/predict/{student_id}", response_model=PredictionResponse)
def run_student_prediction(
    student_id: str,
    term: str = Query("Term 1", description="Term to run prediction for"),
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db),
):
    """
    Runs multi-model ML inference and SHAP explainability for an individual student.
    Enforces ownership: teachers can only run predictions for their assigned class students.
    """
    verify_student_access(student_id, current_user, db)
    try:
        result = predict_student(student_id=student_id, term=term, db=db, persist=True)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")


@router.post("/predict-cohort/{class_id}", response_model=CohortPredictionResponse)
def run_cohort_prediction(
    class_id: str,
    term: str = Query("Term 1", description="Term to run cohort predictions for"),
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db),
):
    """
    Runs batch predictions across all students in a class.
    Teachers can only predict for their own class; Admins can predict for any class.
    """
    target_class = db.query(Class).filter(Class.id == class_id).first()
    if not target_class:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    if current_user.role == "teacher" and target_class.teacher_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: you can only run predictions for your own class",
        )

    try:
        result = predict_cohort(class_id=class_id, term=term, db=db, persist=True)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Cohort prediction error: {str(e)}")


@router.get("/model-info", response_model=ModelInfoResponse)
def get_ml_model_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns active ML model info, performance metrics across all 3 models,
    and global feature importance ranking.
    """
    service = ModelService.get_instance()
    info_record = db.query(MlModelInfo).filter(MlModelInfo.id == "singleton").first()

    metrics = {}
    importance = []
    cluster_names = {}
    trained_at = service.metadata.get("trained_at", "")
    status_str = "ready"

    if info_record:
        status_str = info_record.status
        trained_at = info_record.trained_at.isoformat() if info_record.trained_at else trained_at
        if info_record.metrics_json:
            metrics = json.loads(info_record.metrics_json)
        if info_record.feature_importance_json:
            importance = json.loads(info_record.feature_importance_json)
        if info_record.cluster_names_json:
            cluster_names = json.loads(info_record.cluster_names_json)
    else:
        metrics = {
            "regressor": service.metadata.get("regressor_metrics", {}),
            "classifier": service.metadata.get("classifier_metrics", {}),
            "clustering": service.metadata.get("clustering_metrics", {}),
        }
        importance = service.metadata.get("global_importance", [])
        cluster_names = service.metadata.get("cluster_names", {})

    return {
        "model_version": service.model_version,
        "trained_at": trained_at,
        "status": status_str,
        "metrics": metrics,
        "global_feature_importance": importance,
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
    }
