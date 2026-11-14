import uuid
from typing import TYPE_CHECKING
from sqlalchemy import String, Float, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student


class Extracurricular(Base, TimestampMixin):
    __tablename__ = "extracurriculars"
    __table_args__ = (
        CheckConstraint("hours_per_month >= 0", name="check_activity_hours_non_negative"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    activity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # sports | arts | music | club | volunteering | other
    hours_per_month: Mapped[float] = mapped_column(Float, nullable=False)
    term: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="extracurriculars")
