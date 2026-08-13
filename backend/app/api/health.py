from fastapi import APIRouter, Depends
from sqlmodel import Session, text

from app.database import get_session
from app.schemas import HealthResponse
from app.services.inference import inference_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(session: Session = Depends(get_session)) -> HealthResponse:
    # Database check
    try:
        session.exec(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    model_status = "loaded" if inference_service.is_loaded else "mock"

    return HealthResponse(api="healthy", database=db_status, model=model_status)
