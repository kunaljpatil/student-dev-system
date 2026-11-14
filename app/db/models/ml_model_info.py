from typing import Optional
from datetime import datetime
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class MlModelInfo(Base, TimestampMixin):
    __tablename__ = "ml_model_info"

    id: Mapped[str] = mapped_column(String(50), primary_key=True, default="singleton")
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    feature_importance_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cluster_names_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="ready", nullable=False)
