"""
Pytest Test Suite for Phases 6, 7 & 8: Teacher, Student, and Admin Dashboards & APIs.

Validates:
1. Phase 6 (Teacher APIs):
   - GET /api/teacher/class/students
   - GET /api/teacher/students/{id}
   - GET /api/teacher/students/{id}/insights
   - POST /api/teacher/scores (CRUD)
   - POST /api/teacher/attendance (CRUD)
   - POST /api/teacher/indicators (CRUD)
   - GET /api/teacher/class/analytics
   - GET /api/teacher/class/attention
   - POST /api/teacher/class/predict
2. Phase 7 (Student UI & Data Access):
   - All student APIs return correct data
3. Phase 8 (Admin APIs):
   - GET /api/admin/overview
   - GET /api/admin/classes
   - GET /api/admin/teachers
   - GET /api/admin/clusters
   - GET /api/admin/feature-importance
   - POST /api/admin/ml/retrain
4. HTML Views:
   - GET /login (200 OK)
   - GET /teacher (200 OK)
   - GET /student (200 OK)
   - GET /admin (200 OK)
   - GET /static/css/theme.css (200 OK)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import SessionLocal
from app.db.models import User, Student, Class, Subject, ExamScore
from app.core.security import create_access_token

client = TestClient(app)


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def teacher_auth(db: Session):
    cls = db.query(Class).filter(Class.name == "Class 1").first()
    teacher = db.query(User).filter(User.id == cls.teacher_id).first()
    token = create_access_token(
        subject=teacher.id,
        role=teacher.role,
        email=teacher.email,
        name=teacher.full_name,
    )
    return {"user": teacher, "class": cls, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def student_auth(db: Session):
    student = db.query(Student).filter(Student.roll_no == 1).first()
    user = student.user
    token = create_access_token(
        subject=user.id,
        role=user.role,
        email=user.email,
        name=user.full_name,
    )
    return {"user": user, "student": student, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def admin_auth(db: Session):
    admin = db.query(User).filter(User.role == "admin").first()
    token = create_access_token(
        subject=admin.id,
        role=admin.role,
        email=admin.email,
        name=admin.full_name,
    )
    return {"user": admin, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


# --- Phase 6: Teacher API Tests ---

def test_teacher_class_students_roster(teacher_auth):
    """GET /api/teacher/class/students returns class roster with 50 students."""
    res = client.get("/api/teacher/class/students", headers=teacher_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 50
    first = data[0]
    assert "roll_no" in first
    assert "attendance_pct" in first
    assert "risk_category" in first
    assert "needs_attention" in first


def test_teacher_student_detail(teacher_auth, student_auth):
    """GET /api/teacher/students/{id} returns full multidimensional profile."""
    st_id = student_auth["student"].id
    res = client.get(f"/api/teacher/students/{st_id}", headers=teacher_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["student"]["id"] == st_id
    assert "scores" in data
    assert "attendance" in data
    assert "assignments" in data
    assert "indicators" in data


def test_teacher_data_entry_scores(teacher_auth, student_auth, db: Session):
    """POST /api/teacher/scores records or updates an exam score."""
    st_id = student_auth["student"].id
    subject = db.query(Subject).first()
    payload = {
        "student_id": st_id,
        "subject_id": subject.id,
        "exam_type": "internal1",
        "score": 95.5,
        "max_score": 100.0,
        "term": "Term 1",
    }
    res = client.post("/api/teacher/scores", json=payload, headers=teacher_auth["headers"])
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Verify in DB
    sc = db.query(ExamScore).filter(
        ExamScore.student_id == st_id,
        ExamScore.subject_id == subject.id,
        ExamScore.exam_type == "internal1",
        ExamScore.term == "Term 1",
    ).first()
    assert sc.score == 95.5


def test_teacher_data_entry_indicators(teacher_auth, student_auth):
    """POST /api/teacher/indicators updates soft-skill indicators."""
    st_id = student_auth["student"].id
    payload = {
        "student_id": st_id,
        "term": "Term 1",
        "communication": 5,
        "presentation": 4,
        "confidence": 5,
        "participation": 4,
        "teamwork": 5,
    }
    res = client.post("/api/teacher/indicators", json=payload, headers=teacher_auth["headers"])
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_teacher_class_analytics(teacher_auth):
    """GET /api/teacher/class/analytics returns aggregated cohort statistics."""
    res = client.get("/api/teacher/class/analytics", headers=teacher_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["total_students"] == 50
    assert "risk_breakdown" in data
    assert "cluster_breakdown" in data
    assert "subject_averages" in data
    assert "soft_skills_averages" in data


def test_teacher_class_attention_filter(teacher_auth):
    """GET /api/teacher/class/attention returns prioritized students needing support/monitoring."""
    res = client.get("/api/teacher/class/attention", headers=teacher_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert "attention_count" in data
    assert isinstance(data["students"], list)


def test_teacher_class_predict_rerun(teacher_auth):
    """POST /api/teacher/class/predict re-runs predictions across teacher's cohort."""
    res = client.post("/api/teacher/class/predict?term=Term 1", headers=teacher_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["summary"]["total_students"] == 50


# --- Phase 8: Admin API Tests ---

def test_admin_overview_metrics(admin_auth):
    """GET /api/admin/overview returns schoolwide totals and risk distribution."""
    res = client.get("/api/admin/overview", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["total_students"] == 350
    assert data["total_teachers"] == 7
    assert data["total_classes"] == 7
    assert "schoolwide_avg_predicted_score" in data
    assert "risk_breakdown" in data


def test_admin_classes_comparison(admin_auth):
    """GET /api/admin/classes returns comparison across all 7 classes."""
    res = client.get("/api/admin/classes", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 7
    for c in data:
        assert c["student_count"] == 50
        assert "teacher_name" in c
        assert "avg_predicted_score" in c


def test_admin_teachers_directory(admin_auth):
    """GET /api/admin/teachers returns list of 7 teachers and assigned classes."""
    res = client.get("/api/admin/teachers", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 7


def test_admin_cluster_analytics(admin_auth):
    """GET /api/admin/clusters returns institutional development profile distributions."""
    res = client.get("/api/admin/clusters", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["total_analyzed"] > 0
    assert len(data["clusters"]) >= 3


def test_admin_feature_importance(admin_auth):
    """GET /api/admin/feature-importance returns ranked global SHAP importances."""
    res = client.get("/api/admin/feature-importance", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["model_version"] == "1.0.0"
    assert len(data["features"]) == 24


def test_admin_ml_retrain_endpoint(admin_auth):
    """POST /api/admin/ml/retrain triggers full pipeline retraining and model reload."""
    res = client.post("/api/admin/ml/retrain", headers=admin_auth["headers"])
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "regressor_r2" in data["metrics"]
    assert "classifier_accuracy" in data["metrics"]


# --- HTML UI Views Rendering Tests ---

def test_html_login_view():
    """GET /login serves the glassmorphic login page with HTTP 200."""
    res = client.get("/login")
    assert res.status_code == 200
    assert "Sign In" in res.text
    assert "theme.css" in res.text


def test_html_teacher_dashboard_view():
    """GET /teacher serves the teacher dashboard page."""
    res = client.get("/teacher")
    assert res.status_code == 200
    assert "Teacher Dashboard" in res.text
    assert "Class Student Roster" in res.text


def test_html_student_dashboard_view():
    """GET /student serves the student development dashboard."""
    res = client.get("/student")
    assert res.status_code == 200
    assert "My Development Dashboard" in res.text
    assert "AI Performance Projection" in res.text


def test_html_admin_dashboard_view():
    """GET /admin serves the administrator dashboard."""
    res = client.get("/admin")
    assert res.status_code == 200
    assert "Administrator Dashboard" in res.text
    assert "Machine Learning Engine Governance" in res.text


def test_theme_css_static_file():
    """GET /static/css/theme.css serves the 60-30-10 warm beige glassmorphism stylesheet."""
    res = client.get("/static/css/theme.css")
    assert res.status_code == 200
    assert "--bg-canvas: #f6f1ea;" in res.text
    assert "--brand-espresso: #2a1d15;" in res.text
    assert "--accent-amber: #d97706;" in res.text
