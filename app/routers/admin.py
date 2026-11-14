"""
Admin API Router (spec Section 7, Phase 8).

Endpoints:
- GET  /api/admin/overview
- GET  /api/admin/classes
- GET  /api/admin/classes/{id}
- GET  /api/admin/teachers
- GET  /api/admin/clusters
- GET  /api/admin/feature-importance
- POST /api/admin/ml/retrain
"""

import json
import numpy as np
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    User,
    Class,
    Student,
    MlPrediction,
    MlModelInfo,
    Attendance,
)
from app.core.dependencies import require_admin
from app.ml.predict import ModelService
from scripts.train_all import run_training_pipeline

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/overview")
def get_admin_overview(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Institution overview: totals, schoolwide average, risk category breakdown."""
    total_students = db.query(Student).count()
    total_teachers = db.query(User).filter(User.role == "teacher").count()
    total_classes = db.query(Class).count()

    preds = db.query(MlPrediction).filter(MlPrediction.term == "Term 1").all()
    scores = [p.predicted_score for p in preds]
    avg_score = round(float(np.mean(scores)), 1) if scores else 0.0

    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    for p in preds:
        risk_counts[p.risk_category] = risk_counts.get(p.risk_category, 0) + 1

    # Active model info
    model_info = db.query(MlModelInfo).filter(MlModelInfo.id == "singleton").first()

    return {
        "total_students": total_students,
        "total_teachers": total_teachers,
        "total_classes": total_classes,
        "schoolwide_avg_predicted_score": avg_score,
        "total_predictions": len(preds),
        "risk_breakdown": {
            cat: {
                "count": count,
                "pct": round(count / len(preds) * 100, 1) if preds else 0.0,
            }
            for cat, count in risk_counts.items()
        },
        "model_version": model_info.version if model_info else "1.0.0",
        "last_trained_at": model_info.trained_at.isoformat() if (model_info and model_info.trained_at) else None,
    }


@router.get("/classes")
def get_all_classes(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns comparative table of all classes with teacher, size, average, and attendance."""
    classes = db.query(Class).order_by(Class.name.asc()).all()
    result = []

    for c in classes:
        st_count = len(c.students)
        student_ids = [s.id for s in c.students]
        
        preds = db.query(MlPrediction).filter(
            MlPrediction.student_id.in_(student_ids),
            MlPrediction.term == "Term 1",
        ).all()
        avg_pred = round(float(np.mean([p.predicted_score for p in preds])), 1) if preds else 0.0

        att_records = db.query(Attendance).filter(Attendance.student_id.in_(student_ids)).all()
        tot = sum(a.days_total for a in att_records)
        pres = sum(a.days_present for a in att_records)
        att_pct = round((pres / tot * 100.0) if tot > 0 else 100.0, 1)

        result.append({
            "id": c.id,
            "name": c.name,
            "academic_year": c.academic_year,
            "teacher_id": c.teacher_id,
            "teacher_name": c.teacher.full_name if c.teacher else "Unassigned",
            "teacher_email": c.teacher.email if c.teacher else "",
            "student_count": st_count,
            "avg_predicted_score": avg_pred,
            "avg_attendance_pct": att_pct,
        })

    return result


@router.get("/classes/{id}")
def get_class_drilldown(
    id: str,
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Deep dive into a specific class for the admin view."""
    cls = db.query(Class).filter(Class.id == id).first()
    if not cls:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Class not found")

    student_ids = [s.id for s in cls.students]
    preds = db.query(MlPrediction).filter(
        MlPrediction.student_id.in_(student_ids),
        MlPrediction.term == "Term 1",
    ).all()

    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    cluster_counts = {}
    for p in preds:
        risk_counts[p.risk_category] = risk_counts.get(p.risk_category, 0) + 1
        cname = p.cluster_name or "Steady Progressors"
        cluster_counts[cname] = cluster_counts.get(cname, 0) + 1

    return {
        "id": cls.id,
        "name": cls.name,
        "teacher_name": cls.teacher.full_name if cls.teacher else "Unassigned",
        "student_count": len(cls.students),
        "risk_breakdown": risk_counts,
        "cluster_breakdown": cluster_counts,
    }


@router.get("/teachers")
def get_teachers_list(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List of all faculty members and their assigned classrooms."""
    teachers = db.query(User).filter(User.role == "teacher").order_by(User.full_name.asc()).all()
    result = []
    for t in teachers:
        assigned_cls = db.query(Class).filter(Class.teacher_id == t.id).first()
        result.append({
            "id": t.id,
            "full_name": t.full_name,
            "email": t.email,
            "assigned_class": assigned_cls.name if assigned_cls else "Unassigned",
            "assigned_class_id": assigned_cls.id if assigned_cls else None,
            "is_active": t.is_active,
        })
    return result


@router.get("/clusters")
def get_cohort_cluster_analytics(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns cluster / developmental profile distribution across all students."""
    preds = db.query(MlPrediction).filter(MlPrediction.term == "Term 1").all()
    counts = {}
    for p in preds:
        cname = p.cluster_name or "Steady Progressors"
        counts[cname] = counts.get(cname, 0) + 1

    total = len(preds) if preds else 1
    return {
        "total_analyzed": len(preds),
        "clusters": [
            {
                "name": name,
                "count": count,
                "percentage": round(count / total * 100, 1),
            }
            for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        ],
    }


@router.get("/feature-importance")
def get_global_feature_importance(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Returns ranked SHAP global feature importances for curriculum and systemic insights."""
    service = ModelService.get_instance()
    info_record = db.query(MlModelInfo).filter(MlModelInfo.id == "singleton").first()
    
    if info_record and info_record.feature_importance_json:
        importance = json.loads(info_record.feature_importance_json)
    else:
        importance = service.metadata.get("global_importance", [])

    return {
        "model_version": service.model_version,
        "features": importance,
    }


@router.post("/ml/retrain")
def retrain_models(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Triggers complete ML pipeline retrain, updates models & artifacts, and updates database info."""
    try:
        metadata = run_training_pipeline(seed=2024, model_version="1.0.0")
        ModelService.reload()
        return {
            "status": "success",
            "message": "ML pipeline successfully retrained, serialized, and reloaded.",
            "metrics": {
                "regressor_r2": metadata["regressor_metrics"]["r2"],
                "classifier_accuracy": metadata["classifier_metrics"]["accuracy"],
                "classifier_macro_f1": metadata["classifier_metrics"]["macro_f1"],
                "clustering_silhouette": metadata["clustering_metrics"]["silhouette_score"],
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrain pipeline failed: {str(e)}",
        )
