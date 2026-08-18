from unittest.mock import patch


def test_transcribe_returns_text(client):
    files = {"audio": ("clip.webm", b"fake-audio", "audio/webm")}

    with patch("app.api.voice.transcribe_audio", return_value="how many rocks"):
        resp = client.post("/api/v1/voice/transcribe", files=files)

    assert resp.status_code == 200
    assert resp.json()["text"] == "how many rocks"


def test_transcribe_rejects_empty_audio(client):
    files = {"audio": ("clip.webm", b"", "audio/webm")}
    resp = client.post("/api/v1/voice/transcribe", files=files)
    assert resp.status_code == 400


def test_transcribe_returns_503_when_unconfigured(client):
    files = {"audio": ("clip.webm", b"fake-audio", "audio/webm")}

    with patch("app.api.voice.transcribe_audio", side_effect=RuntimeError("no key")):
        resp = client.post("/api/v1/voice/transcribe", files=files)

    assert resp.status_code == 503


def test_speak_returns_audio(client):
    with patch("app.api.voice.synthesize_speech", return_value=b"mp3-bytes"):
        resp = client.post("/api/v1/voice/speak", json={"text": "hello"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"mp3-bytes"


def test_speak_rejects_blank_text(client):
    resp = client.post("/api/v1/voice/speak", json={"text": "   "})
    assert resp.status_code == 400
