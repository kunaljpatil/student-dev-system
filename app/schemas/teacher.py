"""
Pydantic Schemas for Teacher Dashboard & Data Entry (Phase 6).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date


class CreateScoreRequest(BaseModel):
    student_id: str
    subject_id: str
    exam_type: str = Field(..., description="internal1 | internal2 | midterm | final")
    score: float = Field(..., ge=0, le=100)
    max_score: float = Field(default=100.0, ge=1)
    term: str = Field(..., description="Term 1 | Term 2")


class CreateAttendanceRequest(BaseModel):
    student_id: str
    month: str = Field(..., description="YYYY-MM (e.g. 2024-09)")
    days_present: int = Field(..., ge=0)
    days_total: int = Field(default=22, ge=1)


class CreateAssignmentRequest(BaseModel):
    student_id: str
    subject_id: str
    title: str = Field(..., max_length=200)
    score: float = Field(..., ge=0, le=100)
    submitted_on_time: bool = True
    term: str = "Term 1"


class CreateProjectRequest(BaseModel):
    student_id: str
    title: str = Field(..., max_length=200)
    role: str = Field(default="member", description="member | lead")
    score: float = Field(..., ge=0, le=100)
    term: str = "Term 1"


class CreateExtracurricularRequest(BaseModel):
    student_id: str
    activity_name: str = Field(..., max_length=200)
    category: str = Field(..., description="sports | arts | music | club | volunteering | other")
    hours_per_month: float = Field(..., ge=0)
    term: str = "Term 1"


class CreateCompetitionRequest(BaseModel):
    student_id: str
    name: str = Field(..., max_length=200)
    level: str = Field(..., description="school | district | state | national")
    result: str = Field(..., description="participated | finalist | winner")
    date_str: Optional[str] = None


class CreateIndicatorRequest(BaseModel):
    student_id: str
    term: str = "Term 1"
    communication: int = Field(..., ge=1, le=5)
    presentation: int = Field(..., ge=1, le=5)
    confidence: int = Field(..., ge=1, le=5)
    participation: int = Field(..., ge=1, le=5)
    teamwork: int = Field(..., ge=1, le=5)


class TeacherStudentRosterItem(BaseModel):
    id: str
    roll_no: int
    full_name: str
    email: str
    attendance_pct: float
    avg_score: float
    predicted_score: Optional[float] = None
    risk_category: str
    cluster_name: str
    needs_attention: bool
