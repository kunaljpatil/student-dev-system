import uuid
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, Boolean, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student
    from app.db.models.subject import Subject


class Assignment(Base, TimestampMixin):
    __tablename__ = "assignments"
    __table_args__ = (
        CheckConstraint("score >= 0 AND score <= 100", name="check_assignment_score_range"),
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
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    submitted_on_time: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    term: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="assignments")
    subject: Mapped["Subject"] = relationship("Subject", back_populates="assignments")
