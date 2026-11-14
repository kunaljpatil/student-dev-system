"""
Configuration settings for AI-Based Student Performance & Development System.
Uses pydantic-settings to validate and load from environment variables.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Based Student Performance & Development System"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    ENV: str = "development"

    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Database URL: default SQLite, or PostgreSQL
    DATABASE_URL: str = "sqlite:///./student_dev.db"

    # Security & JWT
    JWT_SECRET: str = "super-secret-key-change-in-production-min-32-chars-long"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Demo credentials
    DEMO_ADMIN_PASSWORD: str = "Admin@123"
    DEMO_TEACHER_PASSWORD: str = "Teach@123"
    DEMO_STUDENT_PASSWORD: str = "Study@123"

    # CORS origins
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
