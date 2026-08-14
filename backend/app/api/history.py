from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.database import get_session
from app.schemas import PredictionListItem, PredictionListResponse
from app.services.prediction_service import get_prediction, list_predictions

router = APIRouter(prefix="/api/v1", tags=["history"])


@router.get("/predictions", response_model=PredictionListResponse)
def get_predictions(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> PredictionListResponse:
    items, total = list_predictions(session, limit=limit, offset=offset)
    return PredictionListResponse(
        total=total,
        items=[PredictionListItem(**item.model_dump()) for item in items],
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionListItem)
def get_prediction_by_id(
    prediction_id: int, session: Session = Depends(get_session)
) -> PredictionListItem:
    prediction = get_prediction(session, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionListItem(**prediction.model_dump())
