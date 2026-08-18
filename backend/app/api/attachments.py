from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.config import get_settings
from app.services.attachments import cleanup_old_attachments, save_attachment

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
settings = get_settings()


class AttachmentResponse(BaseModel):
    path: str
    filename: str


@router.post("/attachments", response_model=AttachmentResponse)
async def upload_attachment(image: UploadFile = File(...)) -> AttachmentResponse:
    """Store a chat image so the agent can classify it by path."""
    if image.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {image.content_type}",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {settings.MAX_UPLOAD_MB}MB limit",
        )

    # Sweep on write so the temp dir cannot grow without bound.
    cleanup_old_attachments()

    filename = image.filename or "upload"
    saved = save_attachment(data, filename)
    return AttachmentResponse(path=str(saved), filename=filename)
