"""
Pydantic Schemas for ML predictions, explanations, and model info.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class ShapFactor(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float


class ShapData(BaseModel):
    base_value: float
    predicted_score: float
    top_positive: List[ShapFactor]
    top_negative: List[ShapFactor]


class RecommendationItem(BaseModel):
    category: str
    text: str
    priority: int


class PredictionResponse(BaseModel):
    student_id: str
    student_name: str
    roll_number: str
    class_id: str
    term: str
    predicted_score: float
    risk_category: str
    probabilities: Dict[str, float]
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    shap: Dict[str, Any]
    recommendations: List[RecommendationItem]
    model_version: str
    generated_at: str
    prediction_id: Optional[str] = None


class StudentCohortSummary(BaseModel):
    student_id: str
    student_name: str
    roll_number: str
    predicted_score: float
    risk_category: str
    cluster_name: str
    top_positive: List[Dict[str, Any]]
    top_negative: List[Dict[str, Any]]
    recommendation_count: int


class CohortPredictionResponse(BaseModel):
    class_id: str
    term: str
    total_students: int
    average_predicted_score: float
    risk_distribution: Dict[str, Dict[str, Any]]
    cluster_distribution: Dict[str, Dict[str, Any]]
    students: List[StudentCohortSummary]


class FeatureImportanceItem(BaseModel):
    feature: str
    label: str
    importance: float


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    status: str
    metrics: Dict[str, Any]
    global_feature_importance: List[FeatureImportanceItem]
    cluster_names: Dict[str, str]
