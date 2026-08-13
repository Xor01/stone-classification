"""Inference service.

MOCK IMPLEMENTATION — the CV teammate will replace `predict()` with real
model loading (torch/onnx) + preprocessing. Everything else in the backend
(API, DB persistence, agent tools) is already wired against this interface,
so swapping the internals here is the ONLY change needed once the real
model artifact (models/model.pt + models/labels.json) is ready.

Expected contract (must not change without updating callers):
    predict(image_bytes: bytes) -> InferenceResult
"""

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings

settings = get_settings()


@dataclass
class InferenceResult:
    predicted_class: str
    confidence: float
    top_predictions: list[dict] = field(default_factory=list)
    inference_ms: float = 0.0
    model_version: str = settings.MODEL_VERSION


class InferenceService:
    """Loads the model once at startup and exposes `predict()`.

    Swap the body of `_load_model` and `predict` for real torch/onnx
    inference. Keep the public interface identical.
    """

    def __init__(self) -> None:
        self.model_version = settings.MODEL_VERSION
        self.labels: list[str] = self._load_labels()
        self.model = self._load_model()

    def _load_labels(self) -> list[str]:
        labels_path = Path(settings.LABELS_PATH)
        if labels_path.exists():
            with open(labels_path) as f:
                mapping = json.load(f)
            # labels.json format: {"0": "cat", "1": "dog", ...}
            return [mapping[str(i)] for i in range(len(mapping))]
        # Fallback placeholder classes until the real labels.json exists
        return ["class_a", "class_b", "class_c"]

    def _load_model(self):
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            # No trained model yet -> mock mode. Replace this block with:
            #   import torch
            #   model = torch.load(model_path, map_location="cpu")
            #   model.eval()
            #   return model
            return None
        # Placeholder for real model loading once model.pt exists.
        return None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> InferenceResult:
        start = time.perf_counter()

        if self.model is None:
            # Mock inference: deterministic-ish random scores so the rest
            # of the stack (DB, frontend, agent) can be built/tested now.
            scores = [random.random() for _ in self.labels]
            total = sum(scores)
            probs = [s / total for s in scores]
        else:
            # Real inference goes here, e.g.:
            #   tensor = preprocess(image_bytes)
            #   with torch.no_grad():
            #       logits = self.model(tensor)
            #       probs = torch.softmax(logits, dim=1)[0].tolist()
            raise NotImplementedError("Wire real model inference here")

        ranked = sorted(zip(self.labels, probs), key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return InferenceResult(
            predicted_class=ranked[0][0],
            confidence=round(ranked[0][1], 4),
            top_predictions=[
                {"class_name": cls, "probability": round(p, 4)}
                for cls, p in ranked[:5]
            ],
            inference_ms=round(elapsed_ms, 2),
            model_version=self.model_version,
        )


# Singleton, loaded once at import time (mirrors "load once, not per request")
inference_service = InferenceService()
