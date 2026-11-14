import uuid
from typing import Optional, TYPE_CHECKING
from sqlalchemy import String, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.user import User
    from app.db.models.school_class import Class


class Announcement(Base, TimestampMixin):
    __tablename__ = "announcements"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Priority: "normal", "important", "urgent"
    priority: Mapped[str] = mapped_column(String(20), default="normal", nullable=False)
    
    # Target audience: "all", "students", "teachers", "class"
    target_audience: Mapped[str] = mapped_column(String(20), default="all", nullable=False)
    
    # Optional target class (e.g. for teacher announcing to their class or admin targeting a class)
    class_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("classes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    
    # Creator
    created_by_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationships
    author: Mapped["User"] = relationship("User", foreign_keys=[created_by_id])
    target_class: Mapped[Optional["Class"]] = relationship("Class", foreign_keys=[class_id])
