from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.database import get_session
from app.schemas import StatsResponse
from app.services.prediction_service import get_statistics

router = APIRouter(prefix="/api/v1", tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def stats(session: Session = Depends(get_session)) -> StatsResponse:
    return StatsResponse(**get_statistics(session))
