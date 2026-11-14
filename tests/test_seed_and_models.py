import pytest
from app.db.session import SessionLocal
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
)
from app.core.security import verify_password
from app.core.config import settings


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    yield session
    session.close()


def test_seed_scale_and_users(db):
    total_users = db.query(User).count()
    assert total_users == 358, "Expected 1 admin + 7 teachers + 350 students"

    admin = db.query(User).filter_by(role="admin").first()
    assert admin is not None
    assert admin.email == "admin@school.test"
    assert verify_password(settings.DEMO_ADMIN_PASSWORD, admin.password_hash)

    teachers = db.query(User).filter_by(role="teacher").all()
    assert len(teachers) == 7
    for t in teachers:
        assert verify_password(settings.DEMO_TEACHER_PASSWORD, t.password_hash)

    students_users = db.query(User).filter_by(role="student").all()
    assert len(students_users) == 350
    sample_student_user = students_users[0]
    assert verify_password(settings.DEMO_STUDENT_PASSWORD, sample_student_user.password_hash)


def test_classes_and_teachers(db):
    classes = db.query(Class).order_by(Class.name).all()
    assert len(classes) == 7

    for i, c in enumerate(classes):
        assert c.name == f"Class {i + 1}"
        assert c.academic_year == "2024-25"
        assert c.teacher is not None
        assert c.teacher.role == "teacher"
        assert len(c.students) == 50


def test_subjects(db):
    subjects = db.query(Subject).all()
    assert len(subjects) == 5
    names = {s.name for s in subjects}
    expected = {"Math", "Science", "English", "Social Studies", "Computer"}
    assert names == expected


def test_student_multidimensional_records(db):
    total_students = db.query(Student).count()
    assert total_students == 350

    # Pick a sample student and verify full relationships
    student = db.query(Student).first()
    assert student is not None
    assert student.user is not None
    assert student.school_class is not None

    # Check student has exam scores, attendance, assignments, and indicators
    scores = db.query(ExamScore).filter_by(student_id=student.id).all()
    assert len(scores) >= 30, "Expected at least 30 exam scores across 2 terms"

    attendance = db.query(Attendance).filter_by(student_id=student.id).all()
    assert len(attendance) == 8, "Expected 8 monthly attendance records (4 per term)"

    assignments = db.query(Assignment).filter_by(student_id=student.id).all()
    assert len(assignments) == 20, "Expected 20 assignments (2 per subject per term)"

    indicators = db.query(DevelopmentIndicator).filter_by(student_id=student.id).all()
    assert len(indicators) == 2, "Expected 2 indicator records (Term 1 & Term 2)"
    for ind in indicators:
        assert 1 <= ind.communication <= 5
        assert 1 <= ind.presentation <= 5
        assert 1 <= ind.confidence <= 5
        assert 1 <= ind.participation <= 5
        assert 1 <= ind.teamwork <= 5
