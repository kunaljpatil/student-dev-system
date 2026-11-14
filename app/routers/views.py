"""
HTML Views & Template Rendering Router (spec Section 2, Section 8).

Serves server-rendered Jinja2 templates for:
- /login : Authentication portal
- /teacher : Teacher classroom dashboard
- /teacher/students/{id} : Individual student insights & data history
- /student : Student development dashboard
- /admin : Institutional administration & model governance
"""

import os
from pathlib import Path
from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings

templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

router = APIRouter(include_in_schema=False)


@router.get("/login", response_class=HTMLResponse)
def login_view(request: Request):
    """Renders the login portal."""
    return templates.TemplateResponse(request=request, name="login.html")




@router.get("/teacher", response_class=HTMLResponse)
def teacher_dashboard_view(request: Request):
    """Renders the teacher classroom dashboard."""
    return templates.TemplateResponse(request=request, name="teacher/dashboard.html")


@router.get("/teacher/students/{id}", response_class=HTMLResponse)
def teacher_student_detail_view(request: Request, id: str):
    """Renders the teacher individual student detail & insights view."""
    return templates.TemplateResponse(
        request=request,
        name="teacher/student_detail.html",
        context={"student_id": id},
    )


@router.get("/student", response_class=HTMLResponse)
def student_dashboard_view(request: Request):
    """Renders the student personal development dashboard."""
    return templates.TemplateResponse(request=request, name="student/dashboard.html")


@router.get("/admin", response_class=HTMLResponse)
def admin_dashboard_view(request: Request):
    """Renders the central institutional administrator dashboard."""
    return templates.TemplateResponse(request=request, name="admin/dashboard.html")
