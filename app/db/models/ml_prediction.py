import uuid
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime
from sqlalchemy import String, Float, Integer, Text, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.student import Student
    from app.db.models.recommendation import Recommendation


class MlPrediction(Base, TimestampMixin):
    __tablename__ = "ml_predictions"
    __table_args__ = (
        UniqueConstraint("student_id", "term", name="uq_student_term_prediction"),
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
    predicted_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(50), nullable=False)  # On Track | Needs Monitoring | Needs Support
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cluster_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    shap_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    student: Mapped["Student"] = relationship("Student", back_populates="predictions")
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="prediction", cascade="all, delete-orphan"
    )
