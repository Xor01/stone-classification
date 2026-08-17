"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


# ---------- Shared ----------


class TopKPrediction(BaseModel):
    class_name: str
    probability: float


# ---------- /health ----------


class HealthResponse(BaseModel):
    api: str = "healthy"
    database: str
    model: str


# ---------- /api/v1/predict ----------


class PredictionResponse(BaseModel):
    id: int
    predicted_class: str
    confidence: float
    top_predictions: list[TopKPrediction]
    inference_ms: float
    model_version: str
    request_id: str
    created_at: datetime


# ---------- /api/v1/predictions ----------


class PredictionListItem(BaseModel):
    id: int
    image_name: str
    predicted_class: str
    confidence: float
    inference_ms: float
    model_version: str
    created_at: datetime


class PredictionListResponse(BaseModel):
    total: int
    items: list[PredictionListItem]


# ---------- /api/v1/stats ----------


class StatsResponse(BaseModel):
    total_predictions: int
    class_distribution: dict[str, int]
    average_confidence: float | None = None
    average_inference_ms: float | None = None


# ---------- /api/v1/model ----------


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    classes: list[str]
    input_size: list[int] = Field(default_factory=lambda: [224, 224])
    metrics: dict | None = None
    deployment_status: str = "loaded"


# ---------- Errors ----------


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
