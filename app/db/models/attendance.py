import uuid
from typing import TYPE_CHECKING
from sqlalchemy import String, Integer, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student


class Attendance(Base, TimestampMixin):
    __tablename__ = "attendance"
    __table_args__ = (
        UniqueConstraint("student_id", "month", name="uq_student_month_attendance"),
        CheckConstraint("days_present >= 0 AND days_present <= days_total", name="check_attendance_days"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    month: Mapped[str] = mapped_column(String(7), nullable=False)  # YYYY-MM
    days_present: Mapped[int] = mapped_column(Integer, nullable=False)
    days_total: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="attendance")
