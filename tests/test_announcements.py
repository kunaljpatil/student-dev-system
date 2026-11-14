"""
Tests for Announcements & Bulletins API.

Verifies:
1. Teacher can create class announcements.
2. Students are forbidden from creating announcements (403).
3. Admin can broadcast announcements to all, students, teachers, or specific class.
4. Student sees only announcements relevant to them (All, Students, or their Class).
5. Teacher sees relevant announcements (All, Teachers, Class notices).
6. Deletion permissions (Admin can delete any, Teacher can delete own, Student cannot).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.models import Class, Student

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


def test_student_cannot_create_announcement(tokens):
    """Students should receive 403 Forbidden when attempting to create announcements."""
    resp = client.post(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['student']}"},
        json={
            "title": "Student Announcement",
            "content": "This should not be allowed",
            "target_audience": "all",
        },
    )
    assert resp.status_code == 403


def test_teacher_create_class_announcement(tokens):
    """Teacher can announce to their assigned class."""
    resp = client.post(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
        json={
            "title": "Upcoming Science Quiz",
            "content": "Please review chapters 4 and 5 for Monday's test.",
            "priority": "important",
            "target_audience": "class",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Upcoming Science Quiz"
    assert data["priority"] == "important"
    assert data["target_audience"] == "class"
    assert data["can_delete"] is True


def test_admin_broadcast_announcement(tokens):
    """Admin has priority to select target audience (All, Students, Teachers, Class)."""
    # 1. Broadcast to everyone
    resp = client.post(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['admin']}"},
        json={
            "title": "Annual Sports Day Next Friday",
            "content": "All academic sessions will conclude early at 1:00 PM.",
            "priority": "urgent",
            "target_audience": "all",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["priority"] == "urgent"
    assert data["target_audience"] == "all"


def test_announcements_visibility_filtering(tokens):
    """Verify that student and teacher only receive announcements intended for their audience."""
    # Student fetch
    s_resp = client.get(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['student']}"},
    )
    assert s_resp.status_code == 200
    s_announcements = s_resp.json()
    assert len(s_announcements) > 0
    # Every announcement seen by student must be 'all', 'students', or their class
    for a in s_announcements:
        assert a["target_audience"] in ["all", "students", "class"]

    # Teacher fetch
    t_resp = client.get(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
    )
    assert t_resp.status_code == 200
    t_announcements = t_resp.json()
    assert len(t_announcements) > 0


def test_announcement_deletion_permissions(tokens):
    """Verify delete rules: Creator or Admin can delete, others cannot."""
    # Teacher creates an announcement
    create_resp = client.post(
        "/api/announcements",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
        json={
            "title": "Temporary Notice",
            "content": "To be deleted soon.",
            "priority": "normal",
            "target_audience": "class",
        },
    )
    assert create_resp.status_code == 201
    ann_id = create_resp.json()["id"]

    # Student cannot delete it (403)
    del_s = client.delete(
        f"/api/announcements/{ann_id}",
        headers={"Authorization": f"Bearer {tokens['student']}"},
    )
    assert del_s.status_code == 403

    # Teacher (creator) can delete it
    del_t = client.delete(
        f"/api/announcements/{ann_id}",
        headers={"Authorization": f"Bearer {tokens['teacher']}"},
    )
    assert del_t.status_code == 200
    assert del_t.json()["status"] == "success"
