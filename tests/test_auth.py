import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends, status
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import SessionLocal, get_db
from app.db.models import User, Student
from app.core.config import settings
from app.core.dependencies import (
    require_admin,
    require_teacher,
    require_student,
    verify_student_access,
    get_current_user,
)

client = TestClient(app)

# Helper test sub-app to test role and ownership guards in isolation
guard_test_app = FastAPI()


@guard_test_app.get("/test/admin-only")
def admin_only_route(user: User = Depends(require_admin)):
    return {"message": "hello admin", "user": user.email}


@guard_test_app.get("/test/teacher-only")
def teacher_only_route(user: User = Depends(require_teacher)):
    return {"message": "hello teacher", "user": user.email}


@guard_test_app.get("/test/student-only")
def student_only_route(user: User = Depends(require_student)):
    return {"message": "hello student", "user": user.email}


@guard_test_app.get("/test/student/{student_id}")
def student_ownership_route(
    student_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    st = verify_student_access(student_id, user, db)
    return {"student_name": st.full_name, "roll_no": st.roll_no}


guard_client = TestClient(guard_test_app)


@pytest.fixture(scope="module")
def tokens():
    """Logs in admin, teacher1, teacher2, and student001 to get test JWT tokens."""
    # Admin
    r_admin = client.post(
        "/api/auth/login",
        json={"email": "admin@school.test", "password": settings.DEMO_ADMIN_PASSWORD},
    )
    assert r_admin.status_code == 200
    token_admin = r_admin.json()["access_token"]

    # Teacher 1 (Class 1)
    r_t1 = client.post(
        "/api/auth/login",
        json={"email": "teacher1@school.test", "password": settings.DEMO_TEACHER_PASSWORD},
    )
    assert r_t1.status_code == 200
    token_t1 = r_t1.json()["access_token"]

    # Teacher 2 (Class 2)
    r_t2 = client.post(
        "/api/auth/login",
        json={"email": "teacher2@school.test", "password": settings.DEMO_TEACHER_PASSWORD},
    )
    assert r_t2.status_code == 200
    token_t2 = r_t2.json()["access_token"]

    # Student 001 (in Class 1)
    r_s1 = client.post(
        "/api/auth/login",
        json={"email": "student001@school.test", "password": settings.DEMO_STUDENT_PASSWORD},
    )
    assert r_s1.status_code == 200
    token_s1 = r_s1.json()["access_token"]

    # Student 051 (in Class 2)
    r_s51 = client.post(
        "/api/auth/login",
        json={"email": "student051@school.test", "password": settings.DEMO_STUDENT_PASSWORD},
    )
    assert r_s51.status_code == 200
    token_s51 = r_s51.json()["access_token"]

    return {
        "admin": token_admin,
        "teacher1": token_t1,
        "teacher2": token_t2,
        "student1": token_s1,
        "student51": token_s51,
    }


def test_login_endpoints(tokens):
    # Valid logins verified in fixture
    assert tokens["admin"] is not None
    assert tokens["teacher1"] is not None
    assert tokens["student1"] is not None

    # Invalid password
    res = client.post(
        "/api/auth/login",
        json={"email": "admin@school.test", "password": "WrongPassword123"},
    )
    assert res.status_code == 401
    assert "Incorrect email or password" in res.json()["detail"]

    # Non-existent user
    res = client.post(
        "/api/auth/login",
        json={"email": "ghost@school.test", "password": "AnyPassword"},
    )
    assert res.status_code == 401


def test_auth_me_endpoint(tokens):
    # Admin /me
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens['admin']}"})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "admin"
    assert data["email"] == "admin@school.test"
    assert data["teacher_class"] is None

    # Teacher /me (Class 1)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens['teacher1']}"})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "teacher"
    assert data["teacher_class"] is not None
    assert data["teacher_class"]["name"] == "Class 1"

    # Student /me (Roll No 1, Class 1)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens['student1']}"})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "student"
    assert data["student_profile"] is not None
    assert data["student_profile"]["roll_no"] == 1
    assert data["student_profile"]["class_name"] == "Class 1"

    # Missing token
    r = client.get("/api/auth/me")
    assert r.status_code == 401

    # Malformed token
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-valid-token"})
    assert r.status_code == 401


def test_role_guards(tokens):
    headers_admin = {"Authorization": f"Bearer {tokens['admin']}"}
    headers_teacher = {"Authorization": f"Bearer {tokens['teacher1']}"}
    headers_student = {"Authorization": f"Bearer {tokens['student1']}"}

    # Admin access
    assert guard_client.get("/test/admin-only", headers=headers_admin).status_code == 200
    assert guard_client.get("/test/admin-only", headers=headers_teacher).status_code == 403
    assert guard_client.get("/test/admin-only", headers=headers_student).status_code == 403

    # Teacher access
    assert guard_client.get("/test/teacher-only", headers=headers_teacher).status_code == 200
    assert guard_client.get("/test/teacher-only", headers=headers_student).status_code == 403

    # Student access
    assert guard_client.get("/test/student-only", headers=headers_student).status_code == 200
    assert guard_client.get("/test/student-only", headers=headers_teacher).status_code == 403


def test_ownership_guards(tokens):
    db = SessionLocal()
    st1 = db.query(Student).filter_by(roll_no=1).first()  # In Class 1
    # Student in Class 2
    st51 = db.query(Student).filter(Student.roll_no == 1, Student.class_id != st1.class_id).first()
    db.close()

    headers_admin = {"Authorization": f"Bearer {tokens['admin']}"}
    headers_t1 = {"Authorization": f"Bearer {tokens['teacher1']}"}  # Teacher of Class 1
    headers_t2 = {"Authorization": f"Bearer {tokens['teacher2']}"}  # Teacher of Class 2
    headers_s1 = {"Authorization": f"Bearer {tokens['student1']}"}  # Student 1
    headers_s51 = {"Authorization": f"Bearer {tokens['student51']}"}  # Student 51

    # Admin can access both students
    assert guard_client.get(f"/test/student/{st1.id}", headers=headers_admin).status_code == 200
    assert guard_client.get(f"/test/student/{st51.id}", headers=headers_admin).status_code == 200

    # Teacher 1 can access student in Class 1, but NOT student in Class 2
    assert guard_client.get(f"/test/student/{st1.id}", headers=headers_t1).status_code == 200
    res_t1_forbidden = guard_client.get(f"/test/student/{st51.id}", headers=headers_t1)
    assert res_t1_forbidden.status_code == 403
    assert "own assigned class" in res_t1_forbidden.json()["detail"]

    # Teacher 2 can access student in Class 2, but NOT student in Class 1
    assert guard_client.get(f"/test/student/{st51.id}", headers=headers_t2).status_code == 200
    assert guard_client.get(f"/test/student/{st1.id}", headers=headers_t2).status_code == 403

    # Student 1 can access own record, but NOT another student's record
    assert guard_client.get(f"/test/student/{st1.id}", headers=headers_s1).status_code == 200
    res_s1_forbidden = guard_client.get(f"/test/student/{st51.id}", headers=headers_s1)
    assert res_s1_forbidden.status_code == 403
    assert "students can only access their own record" in res_s1_forbidden.json()["detail"]
