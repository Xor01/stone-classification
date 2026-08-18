from unittest.mock import MagicMock, patch

import pytest

from app.services.speech import synthesize_speech, transcribe_audio


def test_transcribe_returns_text():
    fake = MagicMock()
    fake.audio.transcriptions.create.return_value = MagicMock(text="how many rocks")

    with patch("app.services.speech.get_openai_client", return_value=fake):
        assert transcribe_audio(b"audio", "clip.webm") == "how many rocks"

    kwargs = fake.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "whisper-1"


def test_synthesize_returns_audio_bytes():
    fake = MagicMock()
    fake.audio.speech.create.return_value = MagicMock(content=b"mp3-bytes")

    with patch("app.services.speech.get_openai_client", return_value=fake):
        assert synthesize_speech("hello") == b"mp3-bytes"

    kwargs = fake.audio.speech.create.call_args.kwargs
    assert kwargs["input"] == "hello"


def test_transcribe_raises_when_unconfigured():
    with patch("app.services.speech.get_openai_client", return_value=None):
        with pytest.raises(RuntimeError):
            transcribe_audio(b"audio", "clip.webm")


def test_tracing_failure_does_not_break_transcription():
    fake = MagicMock()
    fake.audio.transcriptions.create.return_value = MagicMock(text="still works")

    with patch("app.services.speech.get_openai_client", return_value=fake), patch(
        "app.observability.get_langfuse_client", side_effect=RuntimeError("langfuse down")
    ):
        assert transcribe_audio(b"audio", "clip.webm") == "still works"
