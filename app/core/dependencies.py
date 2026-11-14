"""
FastAPI Security & Dependency Injection layer (spec Section 4, 9).

Enforces:
- JWT Bearer token authentication
- Role-based access control (Student, Teacher, Admin)
- Ownership enforcement (teachers can only access their own class; students only their own record)
"""

from typing import List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, Class, Student
from app.core.security import decode_access_token

security_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Extracts, decodes and verifies the Bearer JWT token, returning the User object."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(auth.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload["sub"]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user


def require_role(*allowed_roles: str):
    """Dependency factory checking that current user belongs to one of the allowed roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires one of {allowed_roles} role(s)",
            )
        return current_user

    return role_checker


# Role shortcut dependencies
require_admin = require_role("admin")
require_teacher = require_role("teacher")
require_student = require_role("student")
require_staff = require_role("admin", "teacher")


def get_teacher_class(
    current_teacher: User = Depends(require_teacher),
    db: Session = Depends(get_db),
) -> Class:
    """Verifies that the teacher is assigned to a class, returning the Class entity."""
    cls = db.query(Class).filter(Class.teacher_id == current_teacher.id).first()
    if not cls:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No assigned class found for this teacher",
        )
    return cls


def get_current_student(
    current_user: User = Depends(require_student),
    db: Session = Depends(get_db),
) -> Student:
    """Returns the Student entity corresponding to the currently authenticated student user."""
    student = db.query(Student).filter(Student.user_id == current_user.id).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student profile not found",
        )
    return student


def verify_student_access(
    student_id: str,
    current_user: User,
    db: Session,
) -> Student:
    """
    Ownership enforcement function:
    - Admin: can access any student in any class.
    - Teacher: can only access students enrolled in their own class.
    - Student: can only access their own student record.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )

    if current_user.role == "admin":
        return student

    if current_user.role == "teacher":
        teacher_cls = db.query(Class).filter(Class.teacher_id == current_user.id).first()
        if not teacher_cls or student.class_id != teacher_cls.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: you can only view students in your own assigned class",
            )
        return student

    if current_user.role == "student":
        if student.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: students can only access their own record",
            )
        return student

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
