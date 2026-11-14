from fastapi import FastAPI, Depends, Request, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
from pathlib import Path

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db, engine
from app.routers import auth, students, teachers, admin, ml, views, assistant, announcements

# Ensure database tables exist (e.g. newly added announcements)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Based Student Performance & Development System (FastAPI + Jinja2 + ML Engine)",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files setup
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include HTML UI Views Router
app.include_router(views.router)

# Include REST API Routers
app.include_router(auth.router, prefix=f"{settings.API_PREFIX}/auth", tags=["Auth"])
app.include_router(students.router, prefix=settings.API_PREFIX)
app.include_router(teachers.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)
app.include_router(ml.router, prefix=settings.API_PREFIX)
app.include_router(assistant.router, prefix=settings.API_PREFIX)
app.include_router(assistant.router, prefix="/api/ai")
app.include_router(announcements.router, prefix=settings.API_PREFIX)


@app.get("/api/health", tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint verifying application and database connectivity."""
    db_status = "connected"
    try:
        # Executes lightweight query
        db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {str(exc)}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status,
        "environment": settings.ENV,
    }


@app.get("/", tags=["Root"])
def root(request: Request):
    """Welcome page / redirect target with browser content negotiation."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health",
    }
