import uuid
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.student import Student


class Class(Base, TimestampMixin):
    __tablename__ = "classes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    academic_year: Mapped[str] = mapped_column(String(50), default="2024-25", nullable=False)
    teacher_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    teacher: Mapped[Optional["User"]] = relationship(
        "User", back_populates="teacher_classes"
    )
    students: Mapped[List["Student"]] = relationship(
        "Student", back_populates="school_class", cascade="all, delete-orphan"
    )
