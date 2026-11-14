"""
Teacher API Router (spec Section 7, Phase 6).

Endpoints:
- GET  /api/teacher/class/students
- GET  /api/teacher/students/{id}
- GET  /api/teacher/students/{id}/insights
- POST /api/teacher/scores
- POST /api/teacher/attendance
- POST /api/teacher/assignments
- POST /api/teacher/projects
- POST /api/teacher/extracurriculars
- POST /api/teacher/competitions
- POST /api/teacher/indicators
- GET  /api/teacher/class/analytics
- GET  /api/teacher/class/attention
- POST /api/teacher/class/predict
"""

import json
import numpy as np
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    User,
    Class,
    Student,
    Subject,
    ExamScore,
    Attendance,
    Assignment,
    Project,
    Extracurricular,
    Competition,
    DevelopmentIndicator,
    MlPrediction,
    Recommendation,
)
from app.core.dependencies import (
    require_teacher,
    get_teacher_class,
    verify_student_access,
)
from app.schemas.teacher import (
    CreateScoreRequest,
    CreateAttendanceRequest,
    CreateAssignmentRequest,
    CreateProjectRequest,
    CreateExtracurricularRequest,
    CreateCompetitionRequest,
    CreateIndicatorRequest,
    TeacherStudentRosterItem,
)
from app.schemas.student import StudentInsightsResponse
from app.ml.predict import predict_student, predict_cohort

router = APIRouter(prefix="/teacher", tags=["Teacher"])


@router.get("/class/students", response_model=List[TeacherStudentRosterItem])
def get_class_students(
    teacher_class: Class = Depends(get_teacher_class),
    db: Session = Depends(get_db),
):
    """Returns all students in the teacher's class with current status and metrics."""
    students = (
        db.query(Student)
        .filter(Student.class_id == teacher_class.id)
        .order_by(Student.roll_no.asc())
        .all()
    )

    roster = []
    for st in students:
        # Pre-fetch attendance pct
        att_records = db.query(Attendance).filter(Attendance.student_id == st.id).all()
        tot_days = sum(a.days_total for a in att_records)
        pres_days = sum(a.days_present for a in att_records)
        att_pct = round((pres_days / tot_days * 100.0) if tot_days > 0 else 100.0, 1)

        # Pre-fetch avg scores
        exam_scores = db.query(ExamScore).filter(ExamScore.student_id == st.id).all()
        avg_score = round(float(np.mean([s.score for s in exam_scores])), 1) if exam_scores else 60.0

        # Pre-fetch prediction
        pred = (
            db.query(MlPrediction)
            .filter(MlPrediction.student_id == st.id, MlPrediction.term == "Term 1")
            .first()
        )

        risk_cat = pred.risk_category if pred else "On Track"
        cluster_name = pred.cluster_name if (pred and pred.cluster_name) else "Steady Progressors"
        pred_score = pred.predicted_score if pred else None
        needs_attention = risk_cat in ["Needs Support", "Needs Monitoring"] or att_pct < 80.0

        roster.append({
            "id": st.id,
            "roll_no": st.roll_no,
            "full_name": st.user.full_name if st.user else st.full_name,
            "email": st.user.email if st.user else "",
            "attendance_pct": att_pct,
            "avg_score": avg_score,
            "predicted_score": pred_score,
            "risk_category": risk_cat,
            "cluster_name": cluster_name,
            "needs_attention": needs_attention,
        })

    return roster


@router.get("/students/{id}")
def get_student_detail(
    id: str,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    """Returns full multidimensional profile for a specific student in teacher's class."""
    student = verify_student_access(id, current_teacher, db)
    
    cls = db.query(Class).filter(Class.id == student.class_id).first()
    subjects = {s.id: s.name for s in db.query(Subject).all()}

    scores = db.query(ExamScore).filter(ExamScore.student_id == student.id).all()
    attendance = db.query(Attendance).filter(Attendance.student_id == student.id).order_by(Attendance.month.asc()).all()
    assignments = db.query(Assignment).filter(Assignment.student_id == student.id).all()
    projects = db.query(Project).filter(Project.student_id == student.id).all()
    activities = db.query(Extracurricular).filter(Extracurricular.student_id == student.id).all()
    competitions = db.query(Competition).filter(Competition.student_id == student.id).all()
    indicators = db.query(DevelopmentIndicator).filter(DevelopmentIndicator.student_id == student.id).all()
    prediction = db.query(MlPrediction).filter(MlPrediction.student_id == student.id, MlPrediction.term == "Term 1").first()

    return {
        "student": {
            "id": student.id,
            "roll_no": student.roll_no,
            "full_name": student.user.full_name if student.user else student.full_name,
            "email": student.user.email if student.user else "",
            "class_id": student.class_id,
            "class_name": cls.name if cls else "",
            "academic_year": cls.academic_year if cls else "2024-25",
        },
        "scores": [
            {
                "id": s.id,
                "subject_name": subjects.get(s.subject_id, "Subject"),
                "exam_type": s.exam_type,
                "score": s.score,
                "max_score": s.max_score,
                "term": s.term,
            }
            for s in scores
        ],
        "attendance": [
            {
                "month": a.month,
                "days_present": a.days_present,
                "days_total": a.days_total,
                "pct": round((a.days_present / a.days_total * 100), 1) if a.days_total > 0 else 100,
            }
            for a in attendance
        ],
        "assignments": [
            {
                "id": asg.id,
                "subject_name": subjects.get(asg.subject_id, "Subject"),
                "title": asg.title,
                "score": asg.score,
                "submitted_on_time": asg.submitted_on_time,
                "term": asg.term,
            }
            for asg in assignments
        ],
        "projects": [
            {"id": p.id, "title": p.title, "role": p.role, "score": p.score, "term": p.term}
            for p in projects
        ],
        "activities": [
            {"id": act.id, "activity_name": act.activity_name, "category": act.category, "hours": act.hours_per_month, "term": act.term}
            for act in activities
        ],
        "competitions": [
            {"id": c.id, "name": c.name, "level": c.level, "result": c.result, "date": c.date.isoformat() if c.date else ""}
            for c in competitions
        ],
        "indicators": [
            {
                "term": ind.term,
                "communication": ind.communication,
                "presentation": ind.presentation,
                "confidence": ind.confidence,
                "participation": ind.participation,
                "teamwork": ind.teamwork,
            }
            for ind in indicators
        ],
        "has_prediction": prediction is not None,
    }


@router.get("/students/{id}/insights", response_model=StudentInsightsResponse)
def get_student_insights(
    id: str,
    term: str = Query("Term 1"),
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    """Returns predictive analysis, SHAP factors, and recommendations for a student in teacher's class."""
    student = verify_student_access(id, current_teacher, db)
    cls = db.query(Class).filter(Class.id == student.class_id).first()

    pred = db.query(MlPrediction).filter(MlPrediction.student_id == student.id, MlPrediction.term == term).first()
    if not pred:
        # Run on demand
        res = predict_student(student_id=student.id, term=term, db=db, persist=True)
        return {
            "prediction_id": res["prediction_id"],
            "student_id": student.id,
            "student_name": student.user.full_name if student.user else student.full_name,
            "roll_number": str(student.roll_no),
            "class_name": cls.name if cls else "",
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

    shap_data = json.loads(pred.shap_json)
    recs = db.query(Recommendation).filter(Recommendation.prediction_id == pred.id).order_by(Recommendation.priority.asc()).all()
    rec_items = [{"category": r.category, "text": r.text, "priority": r.priority} for r in recs]

    return {
        "prediction_id": pred.id,
        "student_id": student.id,
        "student_name": student.user.full_name if student.user else student.full_name,
        "roll_number": str(student.roll_no),
        "class_name": cls.name if cls else "",
        "term": term,
        "predicted_score": pred.predicted_score,
        "risk_category": pred.risk_category,
        "cluster_id": pred.cluster_id,
        "cluster_name": pred.cluster_name or "Steady Progressors",
        "shap": shap_data,
        "recommendations": rec_items,
        "model_version": pred.model_version,
        "generated_at": pred.generated_at.isoformat() if pred.generated_at else "",
    }


# --- Data Entry Endpoints ---

@router.post("/scores")
def add_exam_score(
    payload: CreateScoreRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    
    # Check duplicate
    existing = db.query(ExamScore).filter(
        ExamScore.student_id == student.id,
        ExamScore.subject_id == payload.subject_id,
        ExamScore.exam_type == payload.exam_type,
        ExamScore.term == payload.term,
    ).first()

    if existing:
        existing.score = payload.score
        existing.max_score = payload.max_score
    else:
        score_rec = ExamScore(
            student_id=student.id,
            subject_id=payload.subject_id,
            exam_type=payload.exam_type,
            score=payload.score,
            max_score=payload.max_score,
            term=payload.term,
        )
        db.add(score_rec)

    db.commit()
    return {"status": "success", "message": "Exam score recorded successfully"}


@router.post("/attendance")
def add_attendance(
    payload: CreateAttendanceRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    
    existing = db.query(Attendance).filter(
        Attendance.student_id == student.id,
        Attendance.month == payload.month,
    ).first()

    if existing:
        existing.days_present = payload.days_present
        existing.days_total = payload.days_total
    else:
        att = Attendance(
            student_id=student.id,
            month=payload.month,
            days_present=payload.days_present,
            days_total=payload.days_total,
        )
        db.add(att)

    db.commit()
    return {"status": "success", "message": "Attendance record updated successfully"}


@router.post("/assignments")
def add_assignment(
    payload: CreateAssignmentRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    asg = Assignment(
        student_id=student.id,
        subject_id=payload.subject_id,
        title=payload.title,
        score=payload.score,
        submitted_on_time=payload.submitted_on_time,
        term=payload.term,
    )
    db.add(asg)
    db.commit()
    return {"status": "success", "message": "Assignment recorded successfully"}


@router.post("/projects")
def add_project(
    payload: CreateProjectRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    proj = Project(
        student_id=student.id,
        title=payload.title,
        role=payload.role,
        score=payload.score,
        term=payload.term,
    )
    db.add(proj)
    db.commit()
    return {"status": "success", "message": "Project record saved successfully"}


@router.post("/extracurriculars")
def add_extracurricular(
    payload: CreateExtracurricularRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    act = Extracurricular(
        student_id=student.id,
        activity_name=payload.activity_name,
        category=payload.category,
        hours_per_month=payload.hours_per_month,
        term=payload.term,
    )
    db.add(act)
    db.commit()
    return {"status": "success", "message": "Extracurricular activity saved successfully"}


@router.post("/competitions")
def add_competition(
    payload: CreateCompetitionRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    comp_date = None
    if payload.date_str:
        try:
            comp_date = datetime.strptime(payload.date_str, "%Y-%m-%d").date()
        except ValueError:
            pass

    comp = Competition(
        student_id=student.id,
        name=payload.name,
        level=payload.level,
        result=payload.result,
        date=comp_date,
    )
    db.add(comp)
    db.commit()
    return {"status": "success", "message": "Competition achievement recorded successfully"}


@router.post("/indicators")
def add_indicator(
    payload: CreateIndicatorRequest,
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
):
    student = verify_student_access(payload.student_id, current_teacher, db)
    existing = db.query(DevelopmentIndicator).filter(
        DevelopmentIndicator.student_id == student.id,
        DevelopmentIndicator.term == payload.term,
    ).first()

    if existing:
        existing.communication = payload.communication
        existing.presentation = payload.presentation
        existing.confidence = payload.confidence
        existing.participation = payload.participation
        existing.teamwork = payload.teamwork
    else:
        ind = DevelopmentIndicator(
            student_id=student.id,
            term=payload.term,
            communication=payload.communication,
            presentation=payload.presentation,
            confidence=payload.confidence,
            participation=payload.participation,
            teamwork=payload.teamwork,
        )
        db.add(ind)

    db.commit()
    return {"status": "success", "message": "Development indicators saved successfully"}


# --- Analytics & Attention Endpoints ---

@router.get("/class/analytics")
def get_class_analytics(
    teacher_class: Class = Depends(get_teacher_class),
    db: Session = Depends(get_db),
):
    """Calculates class analytics: score distribution, attendance mean, risk and cluster shares."""
    students = db.query(Student).filter(Student.class_id == teacher_class.id).all()
    student_ids = [s.id for s in students]

    # Predictions
    preds = db.query(MlPrediction).filter(
        MlPrediction.student_id.in_(student_ids),
        MlPrediction.term == "Term 1",
    ).all()

    scores = [p.predicted_score for p in preds]
    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    cluster_counts = {}

    for p in preds:
        risk_counts[p.risk_category] = risk_counts.get(p.risk_category, 0) + 1
        cname = p.cluster_name or "Steady Progressors"
        cluster_counts[cname] = cluster_counts.get(cname, 0) + 1

    # Attendance overall
    att_records = db.query(Attendance).filter(Attendance.student_id.in_(student_ids)).all()
    tot_days = sum(a.days_total for a in att_records)
    pres_days = sum(a.days_present for a in att_records)
    avg_att = round((pres_days / tot_days * 100.0) if tot_days > 0 else 100.0, 1)

    # Subject performance
    subjects = db.query(Subject).all()
    subject_stats = {}
    for sub in subjects:
        sub_scores = (
            db.query(ExamScore.score)
            .filter(ExamScore.student_id.in_(student_ids), ExamScore.subject_id == sub.id)
            .all()
        )
        vals = [s[0] for s in sub_scores]
        subject_stats[sub.name] = round(float(np.mean(vals)), 1) if vals else 60.0

    # Soft skills averages
    indicators = (
        db.query(DevelopmentIndicator)
        .filter(DevelopmentIndicator.student_id.in_(student_ids), DevelopmentIndicator.term == "Term 1")
        .all()
    )
    soft_skills = {
        "communication": round(float(np.mean([i.communication for i in indicators])), 2) if indicators else 3.0,
        "presentation": round(float(np.mean([i.presentation for i in indicators])), 2) if indicators else 3.0,
        "confidence": round(float(np.mean([i.confidence for i in indicators])), 2) if indicators else 3.0,
        "participation": round(float(np.mean([i.participation for i in indicators])), 2) if indicators else 3.0,
        "teamwork": round(float(np.mean([i.teamwork for i in indicators])), 2) if indicators else 3.0,
    }

    return {
        "class_name": teacher_class.name,
        "total_students": len(students),
        "predicted_count": len(preds),
        "class_avg_predicted_score": round(float(np.mean(scores)), 1) if scores else 0.0,
        "class_avg_attendance_pct": avg_att,
        "risk_breakdown": risk_counts,
        "cluster_breakdown": cluster_counts,
        "subject_averages": subject_stats,
        "soft_skills_averages": soft_skills,
    }


@router.get("/class/attention")
def get_students_needing_attention(
    teacher_class: Class = Depends(get_teacher_class),
    db: Session = Depends(get_db),
):
    """Returns list of students needing support or monitoring in teacher's class."""
    students = db.query(Student).filter(Student.class_id == teacher_class.id).all()
    student_ids = [s.id for s in students]

    preds = (
        db.query(MlPrediction)
        .filter(
            MlPrediction.student_id.in_(student_ids),
            MlPrediction.term == "Term 1",
            MlPrediction.risk_category.in_(["Needs Support", "Needs Monitoring"]),
        )
        .all()
    )

    pred_map = {p.student_id: p for p in preds}
    attention_list = []

    for st in students:
        if st.id in pred_map:
            p = pred_map[st.id]
            # Get recommendations count
            rec_count = db.query(Recommendation).filter(Recommendation.prediction_id == p.id).count()
            attention_list.append({
                "student_id": st.id,
                "roll_no": st.roll_no,
                "full_name": st.user.full_name if st.user else st.full_name,
                "predicted_score": p.predicted_score,
                "risk_category": p.risk_category,
                "cluster_name": p.cluster_name,
                "recommendation_count": rec_count,
            })

    # Sort so Needs Support is first, then lowest predicted score
    attention_list.sort(key=lambda x: (0 if x["risk_category"] == "Needs Support" else 1, x["predicted_score"]))
    return {
        "class_name": teacher_class.name,
        "attention_count": len(attention_list),
        "students": attention_list,
    }


@router.post("/class/predict")
def rerun_class_predictions(
    term: str = Query("Term 1"),
    teacher_class: Class = Depends(get_teacher_class),
    db: Session = Depends(get_db),
):
    """Triggers on-demand re-prediction across all students in teacher's class."""
    res = predict_cohort(class_id=teacher_class.id, term=term, db=db, persist=True)
    return {
        "status": "success",
        "message": f"Successfully updated predictions for all {res['total_students']} students in {teacher_class.name}.",
        "summary": res,
    }
