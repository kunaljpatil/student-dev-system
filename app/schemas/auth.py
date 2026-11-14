from typing import Optional
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255, description="User's email address")
    password: str = Field(..., min_length=1, description="User's plain password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: str
    email: str
    full_name: Optional[str] = None


class ClassBrief(BaseModel):
    id: str
    name: str
    academic_year: str


class StudentProfileBrief(BaseModel):
    id: str
    roll_no: int
    class_id: str
    class_name: str


class UserMeResponse(BaseModel):
    id: str
    email: str
    role: str
    full_name: Optional[str] = None
    is_active: bool
    teacher_class: Optional[ClassBrief] = None
    student_profile: Optional[StudentProfileBrief] = None
