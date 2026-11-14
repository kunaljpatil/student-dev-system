from typing import Optional
from pydantic import BaseModel, Field


class AnnouncementCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Announcement title")
    content: str = Field(..., min_length=1, max_length=2000, description="Announcement body")
    priority: str = Field(default="normal", description="'normal', 'important', or 'urgent'")
    target_audience: str = Field(default="all", description="'all', 'students', 'teachers', or 'class'")
    class_id: Optional[str] = Field(default=None, description="Optional class ID when target_audience is 'class'")


class AnnouncementResponse(BaseModel):
    id: str
    title: str
    content: str
    priority: str
    target_audience: str
    class_id: Optional[str] = None
    class_name: Optional[str] = None
    author_name: str
    author_role: str
    created_at: str
    can_delete: bool = False
