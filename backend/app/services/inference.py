"""Inference service.

Loads the real ConvNeXt-Tiny stone-classification model trained by the CV
teammate (training/train.py) and runs predictions against it. Falls back
to a MOCK mode automatically when models/model.pt is not present yet, so
the rest of the stack (frontend, agent, tests) can keep working before the
real model artifact is dropped in.

Architecture/preprocessing here are copied exactly from the CV teammate's
training/evaluate.py and training/transforms.py to guarantee the same
results at inference time as during training/evaluation:
    - Backbone: torchvision ConvNeXt-Tiny (ImageNet-pretrained), classifier
      head replaced with Dropout(0.3) + Linear(in_features, num_classes).
    - Saved artifact: model.state_dict() only (not the full model object).
    - Preprocessing: Resize((224, 224)) -> ToTensor() -> Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — the
      standard ImageNet stats, no augmentation (this is the validation
      transform, not the training transform).
"""

import io
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
    """Loads the model once at startup and exposes `predict()`."""

    def __init__(self) -> None:
        self.model_version = settings.MODEL_VERSION
        self.labels: list[str] = self._load_labels()
        self.model = self._load_model()
        self._val_transform = None  # built lazily, only if a real model loaded

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
            # No trained model artifact yet -> mock mode.
            return None

        import torch
        import torch.nn as nn
        import torchvision.models as models

        num_classes = len(self.labels)

        # Rebuild the exact same architecture used in training/train.py
        # and training/evaluate.py before loading the saved state_dict —
        # the .pt file only contains weights, not the model definition.
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _get_val_transform(self):
        if self._val_transform is None:
            import torchvision.transforms as T

            # Matches training/transforms.py's val_transform exactly —
            # same resize + normalization, no augmentation.
            self._val_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        return self._val_transform

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
            import torch
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._get_val_transform()(image).unsqueeze(0)  # add batch dim

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0].tolist()

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
