import uuid
from typing import Optional, List, TYPE_CHECKING
from datetime import date
from sqlalchemy import String, Integer, Date, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.school_class import Class
    from app.db.models.exam_score import ExamScore
    from app.db.models.attendance import Attendance
    from app.db.models.assignment import Assignment
    from app.db.models.project import Project
    from app.db.models.extracurricular import Extracurricular
    from app.db.models.competition import Competition
    from app.db.models.development_indicator import DevelopmentIndicator
    from app.db.models.ml_prediction import MlPrediction
    from app.db.models.recommendation import Recommendation


class Student(Base, TimestampMixin):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    class_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("classes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    roll_no: Mapped[int] = mapped_column(Integer, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="student_profile")
    school_class: Mapped["Class"] = relationship("Class", back_populates="students")
    exam_scores: Mapped[List["ExamScore"]] = relationship(
        "ExamScore", back_populates="student", cascade="all, delete-orphan"
    )
    attendance: Mapped[List["Attendance"]] = relationship(
        "Attendance", back_populates="student", cascade="all, delete-orphan"
    )
    assignments: Mapped[List["Assignment"]] = relationship(
        "Assignment", back_populates="student", cascade="all, delete-orphan"
    )
    projects: Mapped[List["Project"]] = relationship(
        "Project", back_populates="student", cascade="all, delete-orphan"
    )
    extracurriculars: Mapped[List["Extracurricular"]] = relationship(
        "Extracurricular", back_populates="student", cascade="all, delete-orphan"
    )
    competitions: Mapped[List["Competition"]] = relationship(
        "Competition", back_populates="student", cascade="all, delete-orphan"
    )
    indicators: Mapped[List["DevelopmentIndicator"]] = relationship(
        "DevelopmentIndicator", back_populates="student", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["MlPrediction"]] = relationship(
        "MlPrediction", back_populates="student", cascade="all, delete-orphan"
    )
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="student", cascade="all, delete-orphan"
    )
