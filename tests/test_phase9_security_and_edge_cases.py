"""
Pytest Test Suite for Phase 9: Security Review & Edge Cases.

Verifies:
1. Password security & bcrypt salting
2. JWT integrity & tampering rejection
3. Role-based privilege escalation prevention
4. Ownership boundaries & cross-class isolation
5. SQL injection safety & ORM parameterization
6. Pydantic input boundary validations
7. Ethical safeguards (zero sensitive demographic attributes)
"""

from jose import jwt
import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import User, Student, Class, Subject
from app.core.security import verify_password, hash_password, create_access_token
from app.ml.features import FEATURE_NAMES

client = TestClient(app)


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def student_one(db: Session):
    st = db.query(Student).filter(Student.roll_no == 1).first()
    token = create_access_token(
        subject=st.user.id,
        role=st.user.role,
        email=st.user.email,
        name=st.user.full_name,
    )
    return {"student": st, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def teacher_class_one(db: Session):
    cls = db.query(Class).filter(Class.name == "Class 1").first()
    teacher = db.query(User).filter(User.id == cls.teacher_id).first()
    token = create_access_token(
        subject=teacher.id,
        role=teacher.role,
        email=teacher.email,
        name=teacher.full_name,
    )
    return {"class": cls, "teacher": teacher, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def student_class_two(db: Session):
    # Find student belonging to Class 2
    cls2 = db.query(Class).filter(Class.name == "Class 2").first()
    st2 = db.query(Student).filter(Student.class_id == cls2.id).first()
    return st2


# --- 1. Password Security Tests ---

def test_bcrypt_hash_uniqueness_and_verification():
    """Verify passwords are salted so identical plaintext passwords produce different hashes."""
    pwd = "SecurePassword@123"
    hash1 = hash_password(pwd)
    hash2 = hash_password(pwd)
    
    assert hash1 != hash2, "Bcrypt hashes should differ due to distinct random salts"
    assert verify_password(pwd, hash1) is True
    assert verify_password(pwd, hash2) is True
    assert verify_password("WrongPassword", hash1) is False


# --- 2. JWT Security & Tampering Tests ---

def test_jwt_tampering_rejected():
    """Verify that tokens with altered payloads or forged signatures are rejected with 401."""
    # Create forged token signed with wrong secret key
    forged_token = jwt.encode(
        {"sub": "fake-admin-id", "role": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        "wrong-secret-key",
        algorithm=settings.JWT_ALGORITHM,
    )
    res = client.get("/api/admin/overview", headers={"Authorization": f"Bearer {forged_token}"})
    assert res.status_code == 401
    assert "Invalid or expired token" in res.json()["detail"]


def test_expired_jwt_rejected():
    """Verify that expired tokens are immediately rejected with 401."""
    expired_token = jwt.encode(
        {"sub": "test-id", "role": "student", "exp": datetime.now(timezone.utc) - timedelta(hours=2)},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    res = client.get("/api/student/me/profile", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401


# --- 3. Privilege Escalation Prevention Tests ---

def test_student_cannot_access_admin_endpoints(student_one):
    """Students cannot access administrative endpoints (403 Forbidden)."""
    res = client.get("/api/admin/overview", headers=student_one["headers"])
    assert res.status_code == 403

    res_retrain = client.post("/api/admin/ml/retrain", headers=student_one["headers"])
    assert res_retrain.status_code == 403


def test_student_cannot_access_teacher_endpoints(student_one):
    """Students cannot access teacher roster or data entry APIs (403 Forbidden)."""
    res = client.get("/api/teacher/class/students", headers=student_one["headers"])
    assert res.status_code == 403

    res_score = client.post("/api/teacher/scores", json={}, headers=student_one["headers"])
    assert res_score.status_code == 403


def test_teacher_cannot_access_admin_endpoints(teacher_class_one):
    """Teachers cannot access institutional administrator endpoints (403 Forbidden)."""
    res = client.get("/api/admin/overview", headers=teacher_class_one["headers"])
    assert res.status_code == 403

    res_retrain = client.post("/api/admin/ml/retrain", headers=teacher_class_one["headers"])
    assert res_retrain.status_code == 403


# --- 4. Ownership Boundaries & Cross-Class Isolation Tests ---

def test_teacher_cannot_view_other_class_student(teacher_class_one, student_class_two):
    """Teacher of Class 1 cannot inspect detail of a student in Class 2 (403 Forbidden)."""
    res = client.get(
        f"/api/teacher/students/{student_class_two.id}",
        headers=teacher_class_one["headers"],
    )
    assert res.status_code == 403
    assert "assigned class" in res.json()["detail"].lower()


def test_teacher_cannot_modify_other_class_student_score(teacher_class_one, student_class_two, db: Session):
    """Teacher of Class 1 cannot submit scores for a student in Class 2 (403 Forbidden)."""
    subj = db.query(Subject).first()
    payload = {
        "student_id": student_class_two.id,
        "subject_id": subj.id,
        "exam_type": "internal1",
        "score": 80.0,
        "max_score": 100.0,
        "term": "Term 1",
    }
    res = client.post("/api/teacher/scores", json=payload, headers=teacher_class_one["headers"])
    assert res.status_code == 403


# --- 5. SQL Injection Resilience Tests ---

def test_sql_injection_attempt_handled_safely(teacher_class_one):
    """Verify SQL injection payloads in parameters are safely escaped by ORM without execution."""
    malicious_id = "'; DROP TABLE students; --"
    res = client.get(
        f"/api/teacher/students/{malicious_id}",
        headers=teacher_class_one["headers"],
    )
    # Must return 404 Not Found cleanly without executing or throwing DB syntax error
    assert res.status_code == 404


# --- 6. Input Boundary Validations ---

def test_invalid_score_boundary_rejected(teacher_class_one, student_one, db: Session):
    """Pydantic rejects scores over 100 or under 0 with 422 Unprocessable Entity."""
    subj = db.query(Subject).first()
    
    # Over 100
    res_high = client.post(
        "/api/teacher/scores",
        json={
            "student_id": student_one["student"].id,
            "subject_id": subj.id,
            "exam_type": "internal1",
            "score": 150.0,
            "term": "Term 1",
        },
        headers=teacher_class_one["headers"],
    )
    assert res_high.status_code == 422

    # Under 0
    res_neg = client.post(
        "/api/teacher/scores",
        json={
            "student_id": student_one["student"].id,
            "subject_id": subj.id,
            "exam_type": "internal1",
            "score": -10.0,
            "term": "Term 1",
        },
        headers=teacher_class_one["headers"],
    )
    assert res_neg.status_code == 422


def test_invalid_indicator_boundary_rejected(teacher_class_one, student_one):
    """Pydantic rejects soft-skill indicators outside [1..5] with 422 Unprocessable Entity."""
    res = client.post(
        "/api/teacher/indicators",
        json={
            "student_id": student_one["student"].id,
            "term": "Term 1",
            "communication": 6,  # Invalid: max is 5
            "presentation": 3,
            "confidence": 3,
            "participation": 3,
            "teamwork": 3,
        },
        headers=teacher_class_one["headers"],
    )
    assert res.status_code == 422


# --- 7. Ethical Safeguards Verification ---

def test_no_sensitive_demographic_attributes_in_feature_pipeline():
    """Verify zero demographic sensitive attributes exist in the machine learning features."""
    forbidden = ["gender", "sex", "caste", "religion", "faith", "income", "wealth", "race", "ethnicity"]
    for feat in FEATURE_NAMES:
        for f in forbidden:
            assert f not in feat.lower(), f"Ethical violation: sensitive feature '{feat}' detected"
