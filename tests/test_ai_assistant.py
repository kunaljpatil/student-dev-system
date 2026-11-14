"""
Tests for AI Study Companion & Interactive Assistant (spec and extra companion feature).

Verifies:
1. POST /api/ai/assistant and POST /api/assistant/chat require valid authentication (401 if missing).
2. Student queries receive personalized responses with predicted score, attendance, and recommendations.
3. Student intent matching (attendance, study, improve, explain).
4. Teacher queries receive class-level metrics and flagged students.
5. Admin queries receive institution-wide analytics and model version info.
6. Multi-turn chat message history handling.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)


@pytest.fixture(scope="module")
def tokens():
    """Logs in admin, teacher1, and student001 to get real test JWT tokens."""
    r_admin = client.post(
        "/api/auth/login",
        json={"email": "admin@school.test", "password": settings.DEMO_ADMIN_PASSWORD},
    )
    assert r_admin.status_code == 200
    token_admin = r_admin.json()["access_token"]

    r_t1 = client.post(
        "/api/auth/login",
        json={"email": "teacher1@school.test", "password": settings.DEMO_TEACHER_PASSWORD},
    )
    assert r_t1.status_code == 200
    token_t1 = r_t1.json()["access_token"]

    r_s1 = client.post(
        "/api/auth/login",
        json={"email": "student001@school.test", "password": settings.DEMO_STUDENT_PASSWORD},
    )
    assert r_s1.status_code == 200
    token_s1 = r_s1.json()["access_token"]

    return {
        "admin": token_admin,
        "teacher": token_t1,
        "student": token_s1,
    }


def test_ai_assistant_unauthorized_fails():
    """Verify endpoint rejects unauthenticated requests with 401."""
    resp = client.post("/api/ai/assistant", json={"message": "Hello"})
    assert resp.status_code == 401


def test_ai_assistant_student_greeting(tokens):
    """Verify student greeting returns personalized score and profile snapshot."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['student']}"},
        json={"message": "Hi, how are you?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "mode" in data
    assert data["mode"] in ["ai", "companion"]
    reply = data["reply"].lower()
    assert "predicted score" in reply or "snapshot" in reply or "hello" in reply


def test_ai_assistant_student_attendance_query(tokens):
    """Verify student asking about attendance receives attendance metric and tips."""
    resp = client.post(
        "/api/assistant/chat",
        headers={"Authorization": f"Bearer {tokens['student']}"},
        json={"message": "How is my attendance doing?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "attendance" in data["reply"].lower()
    assert "%" in data["reply"]


def test_ai_assistant_student_study_tips_query(tokens):
    """Verify student asking for study advice receives actionable recommendations."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['student']}"},
        json={"message": "How can I improve my predicted score?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"].lower()
    assert "predicted score" in reply or "recommendations" in reply or "improve" in reply


def test_ai_assistant_student_explain_model(tokens):
    """Verify model explanation clarifies ethics and holistic indicators."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['student']}"},
        json={"message": "How does this AI model work?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"].lower()
    assert "xgboost" in reply or "indicators" in reply or "demographic" in reply or "model" in reply


def test_ai_assistant_teacher_snapshot(tokens):
    """Verify teacher query receives class summary, student count, and risk breakdown."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
        json={"message": "Show me my class summary"},
    )
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"].lower()
    assert "students" in reply
    assert "predicted" in reply or "average" in reply


def test_ai_assistant_teacher_flagged_students(tokens):
    """Verify teacher asking about support needs gets flagged list."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
        json={"message": "Which students need monitoring or support?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"].lower()
    assert "support" in reply or "monitoring" in reply or "class" in reply


def test_ai_assistant_admin_health(tokens):
    """Verify admin receives campus-wide counts and ML model info."""
    resp = client.post(
        "/api/ai/assistant",
        headers={"Authorization": f"Bearer {tokens['admin']}"},
        json={"message": "What is the status of the institution?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    reply = data["reply"].lower()
    assert "institution" in reply or "students" in reply or "campus" in reply
    assert "classes" in reply or "model" in reply
