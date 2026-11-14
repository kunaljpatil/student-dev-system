import uuid
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Float, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student
    from app.db.models.subject import Subject
    from app.db.models.user import User


class ExamScore(Base, TimestampMixin):
    __tablename__ = "exam_scores"
    __table_args__ = (
        UniqueConstraint(
            "student_id", "subject_id", "exam_type", "term",
            name="uq_student_subject_exam_term"
        ),
        CheckConstraint("score >= 0 AND score <= 100", name="check_score_range"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    subject_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("subjects.id", ondelete="CASCADE"), index=True, nullable=False
    )
    exam_type: Mapped[str] = mapped_column(String(50), nullable=False)  # internal1 | internal2 | midterm | final
    score: Mapped[float] = mapped_column(Float, nullable=False)
    max_score: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    term: Mapped[str] = mapped_column(String(50), nullable=False)  # Term 1 | Term 2
    recorded_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="exam_scores")
    subject: Mapped["Subject"] = relationship("Subject", back_populates="exam_scores")
    recorder: Mapped[Optional["User"]] = relationship("User", back_populates="recorded_scores")
