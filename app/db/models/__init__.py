"""
SQLAlchemy ORM models package.
Imports all models to register them with Base.metadata.
"""

from app.db.base import Base
from app.db.models.user import User
from app.db.models.school_class import Class
from app.db.models.student import Student
from app.db.models.subject import Subject
from app.db.models.exam_score import ExamScore
from app.db.models.attendance import Attendance
from app.db.models.assignment import Assignment
from app.db.models.project import Project
from app.db.models.extracurricular import Extracurricular
from app.db.models.competition import Competition
from app.db.models.development_indicator import DevelopmentIndicator
from app.db.models.ml_prediction import MlPrediction
from app.db.models.recommendation import Recommendation
from app.db.models.ml_model_info import MlModelInfo
from app.db.models.announcement import Announcement

__all__ = [
    "Base",
    "User",
    "Class",
    "Student",
    "Subject",
    "ExamScore",
    "Attendance",
    "Assignment",
    "Project",
    "Extracurricular",
    "Competition",
    "DevelopmentIndicator",
    "MlPrediction",
    "Recommendation",
    "MlModelInfo",
    "Announcement",
]
