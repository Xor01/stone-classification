"""Business logic for running inference and persisting/reading predictions."""

import hashlib

from sqlmodel import Session, func, select

from app.models import Prediction
from app.services.inference import InferenceResult, inference_service


def classify_and_store(
    session: Session, *, image_bytes: bytes, image_name: str
) -> Prediction:
    result: InferenceResult = inference_service.predict(image_bytes)

    prediction = Prediction(
        image_name=image_name,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        inference_ms=result.inference_ms,
        model_version=result.model_version,
        top_k_predictions=result.top_predictions,
        image_hash=hashlib.sha256(image_bytes).hexdigest(),
    )
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    return prediction


def list_predictions(
    session: Session, *, limit: int = 20, offset: int = 0
) -> tuple[list[Prediction], int]:
    total = session.exec(select(func.count()).select_from(Prediction)).one()
    items = session.exec(
        select(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return list(items), total


def get_prediction(session: Session, prediction_id: int) -> Prediction | None:
    return session.get(Prediction, prediction_id)


def get_statistics(session: Session) -> dict:
    total = session.exec(select(func.count()).select_from(Prediction)).one()

    class_rows = session.exec(
        select(Prediction.predicted_class, func.count())
        .group_by(Prediction.predicted_class)
    ).all()
    class_distribution = {cls: count for cls, count in class_rows}

    avg_confidence = session.exec(select(func.avg(Prediction.confidence))).one()
    avg_inference_ms = session.exec(select(func.avg(Prediction.inference_ms))).one()

    return {
        "total_predictions": total,
        "class_distribution": class_distribution,
        "average_confidence": round(avg_confidence, 4) if avg_confidence else None,
        "average_inference_ms": (
            round(avg_inference_ms, 2) if avg_inference_ms else None
        ),
    }
