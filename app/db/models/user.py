import uuid
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student
    from app.db.models.school_class import Class
    from app.db.models.exam_score import ExamScore
    from app.db.models.development_indicator import DevelopmentIndicator


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # student | teacher | admin
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    student_profile: Mapped[Optional["Student"]] = relationship(
        "Student", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    teacher_classes: Mapped[List["Class"]] = relationship(
        "Class", back_populates="teacher"
    )
    recorded_scores: Mapped[List["ExamScore"]] = relationship(
        "ExamScore", back_populates="recorder"
    )
    recorded_indicators: Mapped[List["DevelopmentIndicator"]] = relationship(
        "DevelopmentIndicator", back_populates="recorder"
    )
