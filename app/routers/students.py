"""
Student API Router (spec Section 7).

Endpoints:
- GET /api/student/me/profile
- GET /api/student/me/scores
- GET /api/student/me/attendance
- GET /api/student/me/activities
- GET /api/student/me/insights
"""

import json
import numpy as np
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    Student,
    Class,
    Subject,
    ExamScore,
    Attendance,
    Project,
    Extracurricular,
    Competition,
    MlPrediction,
    Recommendation,
)
from app.core.dependencies import get_current_student
from app.schemas.student import (
    StudentProfileResponse,
    StudentScoresResponse,
    ExamScoreItem,
    StudentAttendanceResponse,
    AttendanceRecordItem,
    StudentActivitiesResponse,
    ProjectItem,
    ExtracurricularItem,
    CompetitionItem,
    StudentInsightsResponse,
)
from app.ml.predict import predict_student

router = APIRouter(prefix="/student", tags=["Student"])


@router.get("/me/profile", response_model=StudentProfileResponse)
def get_my_profile(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Returns the authenticated student's profile information."""
    cls = db.query(Class).filter(Class.id == student.class_id).first()
    return {
        "id": student.id,
        "user_id": student.user_id,
        "full_name": student.user.full_name if student.user else student.full_name,
        "email": student.user.email if student.user else "",
        "roll_number": str(student.roll_no),
        "class_id": student.class_id,
        "class_name": cls.name if cls else "Unknown",
        "academic_year": cls.academic_year if cls else "2024-25",
    }


@router.get("/me/scores", response_model=StudentScoresResponse)
def get_my_scores(
    term: str = Query(None, description="Optional filter by term (e.g. 'Term 1')"),
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Returns all exam scores for the authenticated student."""
    query = db.query(ExamScore).filter(ExamScore.student_id == student.id)
    if term:
        query = query.filter(ExamScore.term == term)
    scores = query.all()

    # Pre-fetch subjects
    subjects = {s.id: s.name for s in db.query(Subject).all()}

    score_items = []
    subject_scores: Dict[str, List[float]] = {}
    all_scores: List[float] = []

    for sc in scores:
        s_name = subjects.get(sc.subject_id, "Subject")
        score_items.append({
            "id": sc.id,
            "subject_id": sc.subject_id,
            "subject_name": s_name,
            "exam_type": sc.exam_type,
            "score": sc.score,
            "max_score": sc.max_score,
            "term": sc.term,
        })
        subject_scores.setdefault(s_name, []).append(sc.score)
        all_scores.append(sc.score)

    subject_averages = {
        name: round(float(np.mean(vals)), 1)
        for name, vals in subject_scores.items()
    }
    overall_avg = round(float(np.mean(all_scores)), 1) if all_scores else 0.0

    return {
        "scores": score_items,
        "subject_averages": subject_averages,
        "overall_average": overall_avg,
    }


@router.get("/me/attendance", response_model=StudentAttendanceResponse)
def get_my_attendance(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Returns monthly attendance records and overall attendance percentage."""
    records = (
        db.query(Attendance)
        .filter(Attendance.student_id == student.id)
        .order_by(Attendance.month.asc())
        .all()
    )

    items = []
    total_present = 0
    total_days = 0

    for rec in records:
        pct = (rec.days_present / rec.days_total * 100.0) if rec.days_total > 0 else 100.0
        items.append({
            "month": rec.month,
            "days_present": rec.days_present,
            "days_total": rec.days_total,
            "percentage": round(pct, 1),
        })
        total_present += rec.days_present
        total_days += rec.days_total

    overall_pct = (total_present / total_days * 100.0) if total_days > 0 else 100.0

    return {
        "records": items,
        "overall_percentage": round(overall_pct, 1),
        "total_days_present": total_present,
        "total_days": total_days,
    }


@router.get("/me/activities", response_model=StudentActivitiesResponse)
def get_my_activities(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Returns projects, extracurricular activities, and competitions."""
    projects = db.query(Project).filter(Project.student_id == student.id).all()
    activities = db.query(Extracurricular).filter(Extracurricular.student_id == student.id).all()
    competitions = db.query(Competition).filter(Competition.student_id == student.id).all()

    return {
        "projects": [
            {
                "id": p.id,
                "title": p.title,
                "role": p.role,
                "score": p.score,
                "term": p.term,
            }
            for p in projects
        ],
        "extracurriculars": [
            {
                "id": a.id,
                "activity_name": a.activity_name,
                "category": a.category,
                "hours_per_month": a.hours_per_month,
                "term": a.term,
            }
            for a in activities
        ],
        "competitions": [
            {
                "id": c.id,
                "name": c.name,
                "level": c.level,
                "result": c.result,
                "date": c.date.isoformat() if c.date else "",
            }
            for c in competitions
        ],
    }


@router.get("/me/insights", response_model=StudentInsightsResponse)
def get_my_insights(
    term: str = Query("Term 1", description="Academic term (e.g. 'Term 1')"),
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Returns the student's ML predictive insights:
    - Predicted overall performance score
    - Risk/support category (On Track, Needs Monitoring, Needs Support)
    - Student Development Profile (Cluster name)
    - Top positive & negative contributing factors (SHAP)
    - Prioritized personalized recommendations (max 5)
    """
    cls = db.query(Class).filter(Class.id == student.class_id).first()
    class_name = cls.name if cls else "Assigned Class"
    student_name = student.user.full_name if student.user else student.full_name

    # Check for existing stored prediction
    prediction = (
        db.query(MlPrediction)
        .filter(MlPrediction.student_id == student.id, MlPrediction.term == term)
        .first()
    )

    if prediction:
        shap_data = json.loads(prediction.shap_json)
        recs = (
            db.query(Recommendation)
            .filter(Recommendation.prediction_id == prediction.id)
            .order_by(Recommendation.priority.asc())
            .all()
        )
        rec_items = [
            {"category": r.category, "text": r.text, "priority": r.priority}
            for r in recs
        ]
        return {
            "prediction_id": prediction.id,
            "student_id": student.id,
            "student_name": student_name,
            "roll_number": str(student.roll_no),
            "class_name": class_name,
            "term": term,
            "predicted_score": prediction.predicted_score,
            "risk_category": prediction.risk_category,
            "cluster_id": prediction.cluster_id,
            "cluster_name": prediction.cluster_name or "Steady Progressors",
            "shap": shap_data,
            "recommendations": rec_items,
            "model_version": prediction.model_version,
            "generated_at": prediction.generated_at.isoformat() if prediction.generated_at else "",
        }

    # If no prediction cached yet, run on-demand inference and persist
    res = predict_student(student_id=student.id, term=term, db=db, persist=True)
    return {
        "prediction_id": res["prediction_id"],
        "student_id": student.id,
        "student_name": student_name,
        "roll_number": str(student.roll_no),
        "class_name": class_name,
        "term": term,
        "predicted_score": res["predicted_score"],
        "risk_category": res["risk_category"],
        "cluster_id": res["cluster_id"],
        "cluster_name": res["cluster_name"],
        "shap": res["shap"],
        "recommendations": res["recommendations"],
        "model_version": res["model_version"],
        "generated_at": res["generated_at"],
    }
