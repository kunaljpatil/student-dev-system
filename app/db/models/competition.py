import uuid
from datetime import date
from typing import TYPE_CHECKING
from sqlalchemy import String, Date, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student


class Competition(Base, TimestampMixin):
    __tablename__ = "competitions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    student_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("students.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    level: Mapped[str] = mapped_column(String(50), nullable=False)  # school | district | state | national
    result: Mapped[str] = mapped_column(String(50), nullable=False)  # participated | finalist | winner
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="competitions")
