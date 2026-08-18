"""Speech-to-text and text-to-speech via OpenAI, traced to Langfuse."""

import io
import logging
from functools import lru_cache

from app.config import get_settings
from app.observability import trace_operation

logger = logging.getLogger("cv-agent-backend")


@lru_cache
def get_openai_client():
    """Build the OpenAI client once, or return None if no key is configured."""
    settings = get_settings()
    key = (settings.OPENAI_API_KEY or "").strip("\"' ")
    if not key:
        logger.warning("OPENAI_API_KEY not set; speech endpoints are unavailable.")
        return None

    from openai import OpenAI

    return OpenAI(api_key=key)


def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Transcribe recorded audio to text. Raises RuntimeError if unconfigured."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    settings = get_settings()
    with trace_operation(
        name="transcribe-audio",
        tags=["cv-backend", "voice"],
        input={"filename": filename, "size_kb": round(len(audio_bytes) / 1024, 1)},
    ) as trace:
        upload = io.BytesIO(audio_bytes)
        upload.name = filename
        result = client.audio.transcriptions.create(
            file=upload,
            model=settings.STT_MODEL,
        )
        text = result.text
        trace.record(output={"text": text}, metadata={"model": settings.STT_MODEL})

    return text


def synthesize_speech(text: str) -> bytes:
    """Render text to speech audio. Raises RuntimeError if unconfigured."""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    settings = get_settings()
    with trace_operation(
        name="synthesize-speech",
        tags=["cv-backend", "voice"],
        input={"characters": len(text)},
    ) as trace:
        result = client.audio.speech.create(
            input=text,
            model=settings.TTS_MODEL,
            voice=settings.TTS_VOICE,
        )
        audio = result.content
        trace.record(
            output={"bytes": len(audio)},
            metadata={"model": settings.TTS_MODEL, "voice": settings.TTS_VOICE},
        )

    return audio
