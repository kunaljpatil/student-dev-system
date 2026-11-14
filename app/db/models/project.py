import uuid
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student


class Project(Base, TimestampMixin):
    __tablename__ = "projects"
    __table_args__ = (
        CheckConstraint("score >= 0 AND score <= 100", name="check_project_score_range"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # member | lead
    score: Mapped[float] = mapped_column(Float, nullable=False)
    term: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="projects")
