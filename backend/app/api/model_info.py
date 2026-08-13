import json
from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import ModelInfoResponse
from app.services.inference import inference_service

router = APIRouter(prefix="/api/v1", tags=["model"])
settings = get_settings()


@router.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metrics = None
    metrics_path = Path("reports/model_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return ModelInfoResponse(
        model_name="cv-agent-classifier",
        version=inference_service.model_version,
        classes=inference_service.labels,
        metrics=metrics,
        deployment_status="loaded" if inference_service.is_loaded else "mock",
    )
