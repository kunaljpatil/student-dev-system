"""
Pydantic Schemas for Student endpoints (profile, scores, attendance, activities, insights).
"""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.schemas.ml import RecommendationItem, ShapData


class StudentProfileResponse(BaseModel):
    id: str
    user_id: str
    full_name: str
    email: str
    roll_number: str
    class_id: str
    class_name: str
    academic_year: str


class ExamScoreItem(BaseModel):
    id: str
    subject_id: str
    subject_name: str
    exam_type: str
    score: float
    max_score: float
    term: str


class StudentScoresResponse(BaseModel):
    scores: List[ExamScoreItem]
    subject_averages: Dict[str, float]
    overall_average: float


class AttendanceRecordItem(BaseModel):
    month: str
    days_present: int
    days_total: int
    percentage: float


class StudentAttendanceResponse(BaseModel):
    records: List[AttendanceRecordItem]
    overall_percentage: float
    total_days_present: int
    total_days: int


class ProjectItem(BaseModel):
    id: str
    title: str
    role: str
    score: float
    term: str


class ExtracurricularItem(BaseModel):
    id: str
    activity_name: str
    category: str
    hours_per_month: float
    term: str


class CompetitionItem(BaseModel):
    id: str
    name: str
    level: str
    result: str
    date: str


class StudentActivitiesResponse(BaseModel):
    projects: List[ProjectItem]
    extracurriculars: List[ExtracurricularItem]
    competitions: List[CompetitionItem]


class StudentInsightsResponse(BaseModel):
    prediction_id: Optional[str] = None
    student_id: str
    student_name: str
    roll_number: str
    class_name: str
    term: str
    predicted_score: float
    risk_category: str
    cluster_id: Optional[int] = None
    cluster_name: str
    shap: Dict[str, Any]
    recommendations: List[RecommendationItem]
    model_version: str
    generated_at: str
    ethics_disclaimer: str = (
        "Notice: Predictive insights and development profiles are algorithmic "
        "decision-support aids designed for constructive student development. "
        "They do not determine permanent academic evaluations."
    )
