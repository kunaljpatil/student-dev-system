import uuid
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Integer, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student
    from app.db.models.user import User


class DevelopmentIndicator(Base, TimestampMixin):
    __tablename__ = "development_indicators"
    __table_args__ = (
        UniqueConstraint("student_id", "term", name="uq_student_term_indicators"),
        CheckConstraint("communication >= 1 AND communication <= 5", name="check_communication_range"),
        CheckConstraint("presentation >= 1 AND presentation <= 5", name="check_presentation_range"),
        CheckConstraint("confidence >= 1 AND confidence <= 5", name="check_confidence_range"),
        CheckConstraint("participation >= 1 AND participation <= 5", name="check_participation_range"),
        CheckConstraint("teamwork >= 1 AND teamwork <= 5", name="check_teamwork_range"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    term: Mapped[str] = mapped_column(String(50), nullable=False)
    communication: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    presentation: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    participation: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    teamwork: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    recorded_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="indicators")
    recorder: Mapped[Optional["User"]] = relationship("User", back_populates="recorded_indicators")
