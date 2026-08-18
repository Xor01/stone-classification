import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.speech import synthesize_speech, transcribe_audio

logger = logging.getLogger("cv-agent-backend")

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


class TranscriptionResponse(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(..., description="Text to render as speech")


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        text = transcribe_audio(data, audio.filename or "clip.webm")
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail="Transcription failed")

    return TranscriptionResponse(text=text)


@router.post("/speak")
async def speak(request: SpeakRequest) -> Response:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be blank")

    try:
        audio = synthesize_speech(request.text)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except Exception as e:
        logger.exception("Speech synthesis failed: %s", e)
        raise HTTPException(status_code=500, detail="Speech synthesis failed")

    return Response(content=audio, media_type="audio/mpeg")
