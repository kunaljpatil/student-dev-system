import uuid
from typing import List, TYPE_CHECKING
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.exam_score import ExamScore
    from app.db.models.assignment import Assignment


class Subject(Base, TimestampMixin):
    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)

    # Relationships
    exam_scores: Mapped[List["ExamScore"]] = relationship(
        "ExamScore", back_populates="subject"
    )
    assignments: Mapped[List["Assignment"]] = relationship(
        "Assignment", back_populates="subject"
    )
