"""
Pytest Test Suite for Phase 5: Recommendation Engine, Prediction Service & ML APIs.

Validates:
1. Recommendation engine rule-based logic (supportive, prioritized, max 5)
2. Prediction service inference & persistence (predict_student, predict_cohort)
3. Student REST APIs (profile, scores, attendance, activities)
4. /insights endpoint returns full payload as mandated by Section 10 Done criterion
5. Role & ownership guards on ML endpoints (teacher own-class, admin access, forbidden cross-class)
6. Model info endpoint
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import SessionLocal
from app.db.models import User, Student, Class, MlPrediction, Recommendation
from app.core.security import create_access_token
from app.ml.recommend import generate_recommendations
from app.ml.predict import predict_student, predict_cohort

client = TestClient(app)


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def student_user(db: Session):
    student = db.query(Student).filter(Student.roll_no == 1).first()
    assert student is not None
    user = student.user
    token = create_access_token(
        subject=user.id,
        role=user.role,
        email=user.email,
        name=user.full_name,
    )
    return {"user": user, "student": student, "token": token}


@pytest.fixture(scope="module")
def teacher_user(db: Session):
    # Teacher 1 teaches Class 1
    cls_1 = db.query(Class).filter(Class.name == "Class 1").first()
    assert cls_1 is not None
    teacher = db.query(User).filter(User.id == cls_1.teacher_id).first()
    token = create_access_token(
        subject=teacher.id,
        role=teacher.role,
        email=teacher.email,
        name=teacher.full_name,
    )
    return {"user": teacher, "class": cls_1, "token": token}


@pytest.fixture(scope="module")
def other_teacher_user(db: Session):
    # Teacher 2 teaches Class 2
    cls_2 = db.query(Class).filter(Class.name == "Class 2").first()
    assert cls_2 is not None
    teacher = db.query(User).filter(User.id == cls_2.teacher_id).first()
    token = create_access_token(
        subject=teacher.id,
        role=teacher.role,
        email=teacher.email,
        name=teacher.full_name,
    )
    return {"user": teacher, "class": cls_2, "token": token}


@pytest.fixture(scope="module")
def admin_user(db: Session):
    admin = db.query(User).filter(User.role == "admin").first()
    token = create_access_token(
        subject=admin.id,
        role=admin.role,
        email=admin.email,
        name=admin.full_name,
    )
    return {"user": admin, "token": token}


# --- 1. Recommendation Engine Unit Tests ---

def test_recommendation_engine_at_risk_student():
    """Verify high-priority support actions triggered for at-risk profiles."""
    features = {
        "attendance_pct": 74.0,
        "min_subject_score": 42.0,
        "avg_internal_score": 45.0,
        "ontime_rate": 0.65,
        "communication": 2.0,
        "presentation": 2.0,
        "participation": 2.0,
        "teamwork": 2.0,
        "activity_hours": 1.0,
    }
    shap_exp = {
        "top_negative": [
            {"feature": "min_subject_score", "shap_value": -6.2},
            {"feature": "attendance_pct", "shap_value": -4.1},
        ]
    }
    recs = generate_recommendations(
        predicted_score=44.5,
        risk_category="Needs Support",
        cluster_name="At Risk - Low Engagement",
        shap_explanation=shap_exp,
        features=features,
    )

    assert 1 <= len(recs) <= 5
    # Verify priority sorting
    priorities = [r["priority"] for r in recs]
    assert priorities == sorted(priorities)
    # Priority 1 items must be present
    assert 1 in priorities
    categories = [r["category"] for r in recs]
    assert "Attendance & Regularity" in categories or "Academic Strengthening" in categories


def test_recommendation_engine_high_achiever():
    """Verify enrichment and leadership actions generated for high achievers."""
    features = {
        "attendance_pct": 96.0,
        "min_subject_score": 82.0,
        "avg_internal_score": 88.0,
        "ontime_rate": 0.98,
        "communication": 4.5,
        "presentation": 4.5,
        "participation": 4.8,
        "teamwork": 4.5,
        "activity_hours": 14.0,
        "competition_count": 2,
        "lead_roles": 1,
    }
    recs = generate_recommendations(
        predicted_score=89.0,
        risk_category="On Track",
        cluster_name="High Achiever - Well Rounded",
        shap_explanation={"top_negative": []},
        features=features,
    )

    assert 1 <= len(recs) <= 5
    categories = [r["category"] for r in recs]
    assert "Advanced Growth & Leadership" in categories or "Enrichment & Special Projects" in categories


# --- 2. Inference Service Tests ---

def test_predict_student_service(db: Session, student_user):
    """Verify end-to-end predict_student inference & database persistence."""
    st = student_user["student"]
    res = predict_student(student_id=st.id, term="Term 1", db=db, persist=True)

    assert res["student_id"] == st.id
    assert 0.0 <= res["predicted_score"] <= 100.0
    assert res["risk_category"] in ["On Track", "Needs Monitoring", "Needs Support"]
    assert res["cluster_name"] in [
        "Steady Progressors",
        "High Achiever - Well Rounded",
        "Academically Strong - Low Engagement",
    ]
    assert "top_positive" in res["shap"]
    assert "top_negative" in res["shap"]
    assert 1 <= len(res["recommendations"]) <= 5
    assert res["model_version"] == "1.0.0"

    # Verify persisted in database
    db_pred = db.query(MlPrediction).filter(
        MlPrediction.student_id == st.id, MlPrediction.term == "Term 1"
    ).first()
    assert db_pred is not None
    assert db_pred.predicted_score == res["predicted_score"]

    db_recs = db.query(Recommendation).filter(
        Recommendation.prediction_id == db_pred.id
    ).all()
    assert len(db_recs) == len(res["recommendations"])


def test_predict_cohort_service(db: Session, teacher_user):
    """Verify batch cohort inference across all students in a class."""
    cls = teacher_user["class"]
    cohort_res = predict_cohort(class_id=cls.id, term="Term 1", db=db, persist=True)

    assert cohort_res["class_id"] == cls.id
    assert cohort_res["total_students"] == 50
    assert cohort_res["average_predicted_score"] > 0
    assert "risk_distribution" in cohort_res
    assert "cluster_distribution" in cohort_res
    assert len(cohort_res["students"]) == 50


# --- 3. Student REST APIs & /insights Deliverable ---

def test_student_profile_api(student_user):
    """Verify GET /api/student/me/profile."""
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/student/me/profile", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["roll_number"] == "1"
    assert data["class_name"] == "Class 1"


def test_student_scores_api(student_user):
    """Verify GET /api/student/me/scores."""
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/student/me/scores", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["scores"]) > 0
    assert "subject_averages" in data
    assert data["overall_average"] > 0


def test_student_attendance_api(student_user):
    """Verify GET /api/student/me/attendance."""
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/student/me/attendance", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["records"]) == 8  # 8 months
    assert 0 <= data["overall_percentage"] <= 100


def test_student_activities_api(student_user):
    """Verify GET /api/student/me/activities."""
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/student/me/activities", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data
    assert "extracurriculars" in data
    assert "competitions" in data


def test_student_insights_full_payload_deliverable(student_user):
    """
    CRITICAL PHASE 5 ACCEPTANCE CRITERIA:
    Verify GET /api/student/me/insights returns full payload.
    """
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/student/me/insights?term=Term 1", headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Required payload items from spec
    assert "predicted_score" in data
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert data["risk_category"] in ["On Track", "Needs Monitoring", "Needs Support"]
    assert "cluster_name" in data
    assert len(data["cluster_name"]) > 3
    assert "shap" in data
    assert "top_positive" in data["shap"]
    assert "top_negative" in data["shap"]
    assert "recommendations" in data
    assert 1 <= len(data["recommendations"]) <= 5
    assert "ethics_disclaimer" in data
    assert data["model_version"] == "1.0.0"


# --- 4. Staff ML APIs & Ownership Enforcement ---

def test_teacher_predict_student_own_class(teacher_user, student_user):
    """Teacher can run predictions for student in their own class."""
    headers = {"Authorization": f"Bearer {teacher_user['token']}"}
    st_id = student_user["student"].id
    response = client.post(f"/api/ml/predict/{st_id}?term=Term 1", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == st_id
    assert "predicted_score" in data


def test_teacher_predict_student_other_class_forbidden(other_teacher_user, student_user):
    """Teacher cannot run predictions for students from another class (403 Forbidden)."""
    headers = {"Authorization": f"Bearer {other_teacher_user['token']}"}
    st_id = student_user["student"].id  # Belongs to Class 10-A, but teacher teaches 10-B
    response = client.post(f"/api/ml/predict/{st_id}?term=Term 1", headers=headers)
    assert response.status_code == 403
    assert "assigned class" in response.json()["detail"].lower()


def test_admin_predict_any_student(admin_user, student_user):
    """Admin has universal permission to run prediction for any student."""
    headers = {"Authorization": f"Bearer {admin_user['token']}"}
    st_id = student_user["student"].id
    response = client.post(f"/api/ml/predict/{st_id}?term=Term 1", headers=headers)
    assert response.status_code == 200


def test_cohort_prediction_ownership(teacher_user, other_teacher_user):
    """Teacher can predict own cohort, but gets 403 on another teacher's class."""
    own_cls_id = teacher_user["class"].id
    # Own class succeeds
    res_own = client.post(
        f"/api/ml/predict-cohort/{own_cls_id}?term=Term 1",
        headers={"Authorization": f"Bearer {teacher_user['token']}"},
    )
    assert res_own.status_code == 200
    assert res_own.json()["total_students"] == 50

    # Cross-class forbidden
    res_cross = client.post(
        f"/api/ml/predict-cohort/{own_cls_id}?term=Term 1",
        headers={"Authorization": f"Bearer {other_teacher_user['token']}"},
    )
    assert res_cross.status_code == 403


def test_get_model_info_api(student_user):
    """Verify GET /api/ml/model-info returns metadata and global feature importances."""
    headers = {"Authorization": f"Bearer {student_user['token']}"}
    response = client.get("/api/ml/model-info", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "1.0.0"
    assert data["status"] == "ready"
    assert "metrics" in data
    assert len(data["global_feature_importance"]) == 24
