from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session

from app.config import get_settings
from app.database import get_session
from app.observability import trace_classification
from app.schemas import PredictionResponse, TopKPrediction
from app.services.prediction_service import classify_and_store

router = APIRouter(prefix="/api/v1", tags=["predictions"])
settings = get_settings()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> PredictionResponse:
    # --- Validation (section 33/34 of the spec) ---
    if image.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {image.content_type}",
        )

    image_bytes = await image.read()

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.MAX_UPLOAD_MB}MB limit",
        )

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        # The span wraps the inference so it records real latency, not 0ms.
        with trace_classification(
            image_name=image.filename or "upload",
            content_type=image.content_type,
            size_bytes=len(image_bytes),
        ) as trace:
            prediction = classify_and_store(
                session, image_bytes=image_bytes, image_name=image.filename or "upload"
            )
            trace.record(prediction)
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model inference is not available yet",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed. Please try again.",
        )

    return PredictionResponse(
        id=prediction.id,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        top_predictions=[
            TopKPrediction(**p) for p in (prediction.top_k_predictions or [])
        ],
        inference_ms=prediction.inference_ms,
        model_version=prediction.model_version,
        request_id=prediction.request_id,
        created_at=prediction.created_at,
    )
