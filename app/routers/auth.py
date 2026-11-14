"""
Authentication Router (spec Section 7).

Endpoints:
- POST /api/auth/login -> JWT Token
- GET  /api/auth/me    -> Current user profile + role info
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import User, Class, Student
from app.core.security import verify_password, create_access_token
from app.core.dependencies import get_current_user
from app.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserMeResponse,
    ClassBrief,
    StudentProfileBrief,
)

router = APIRouter()


@router.post("/login", response_model=TokenResponse, summary="User Login")
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user with email and password, returning a signed JWT access token.
    """
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    token = create_access_token(
        subject=user.id,
        role=user.role,
        email=user.email,
        name=user.full_name,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=UserMeResponse, summary="Current User Profile")
def get_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the currently authenticated user's profile and contextual details
    (assigned class if teacher; student roll no & class if student).
    """
    teacher_class = None
    student_profile = None

    if current_user.role == "teacher":
        cls = db.query(Class).filter(Class.teacher_id == current_user.id).first()
        if cls:
            teacher_class = ClassBrief(
                id=cls.id,
                name=cls.name,
                academic_year=cls.academic_year,
            )

    elif current_user.role == "student":
        st = db.query(Student).filter(Student.user_id == current_user.id).first()
        if st and st.school_class:
            student_profile = StudentProfileBrief(
                id=st.id,
                roll_no=st.roll_no,
                class_id=st.class_id,
                class_name=st.school_class.name,
            )

    return UserMeResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        teacher_class=teacher_class,
        student_profile=student_profile,
    )
