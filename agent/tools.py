import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Canonical copy of this check lives in backend/app/services/attachments.py
# (is_allowed_attachment). agent/ must stay importable standalone (e.g.
# scripts/test_langfuse.py runs it from the repo root where `app` is not on
# sys.path), so we cannot import that module here — keep the two in sync by hand.
ATTACHMENT_DIR = Path(
    os.getenv("ATTACHMENT_DIR") or Path(tempfile.gettempdir()) / "cv-agent-attachments"
)


def _is_allowed_attachment(path: str) -> bool:
    """True only for an existing regular file directly inside ATTACHMENT_DIR."""
    try:
        resolved = Path(path).resolve()
        base = ATTACHMENT_DIR.resolve()
    except Exception:
        return False

    if not resolved.is_file():
        return False
    return resolved.parent == base


@tool
def get_model_info() -> dict:
    """Get information about the currently deployed computer vision model, including model name, version, classes, metrics, and deployment status."""
    response = httpx.get(
        f"{BACKEND_URL}/api/v1/model",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@tool
def get_prediction_history(limit: int = 20, offset: int = 0) -> dict:
    """Get history of predictions made by the computer vision model. Supports pagination via limit and offset."""
    response = httpx.get(
        f"{BACKEND_URL}/api/v1/predictions",
        params={"limit": limit, "offset": offset},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@tool
def get_prediction_by_id(prediction_id: int) -> dict:
    """Get details of a specific prediction by its numeric prediction_id."""
    response = httpx.get(
        f"{BACKEND_URL}/api/v1/predictions/{prediction_id}",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@tool
def get_prediction_statistics() -> dict:
    """Get aggregated statistics of predictions, including total predictions count, class distribution, and average confidence."""
    response = httpx.get(
        f"{BACKEND_URL}/api/v1/stats",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@tool
def classify_image(image_path: str) -> dict:
    """
    Classify an image file using the deployed computer vision model.

    Args:
        image_path: Local filesystem path to the image file to classify.

    Returns:
        The prediction result including predicted class, confidence,
        top predictions, inference time, and model version.
    """
    if not _is_allowed_attachment(image_path):
        return {
            "error": (
                "Refused: image_path must be an existing file inside the "
                "attachment directory. Ask the user to upload the image "
                "through the chat attachment feature first."
            )
        }

    with open(image_path, "rb") as image_file:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/predict",
            files={
                "image": (
                    os.path.basename(image_path),
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=60,
        )

    response.raise_for_status()
    return response.json()


tools = [
    classify_image,
    get_prediction_history,
    get_prediction_by_id,
    get_prediction_statistics,
    get_model_info,
]