"""SQLModel table definitions."""

import uuid
from datetime import datetime, timezone

from sqlmodel import JSON, Column, Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"

    id: int | None = Field(default=None, primary_key=True)

    # Core required fields (per project spec, section 16)
    image_name: str
    predicted_class: str
    confidence: float
    inference_ms: float
    model_version: str
    created_at: datetime = Field(default_factory=_utcnow, nullable=False, index=True)

    # Recommended additional fields
    image_path: str | None = Field(default=None)
    top_k_predictions: list | None = Field(default=None, sa_column=Column(JSON))
    image_hash: str | None = Field(default=None, index=True)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True)
