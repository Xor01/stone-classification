import io
from pathlib import Path

from PIL import Image


def _fake_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_upload_returns_a_usable_path(client):
    files = {"image": ("rock.jpg", _fake_image_bytes(), "image/jpeg")}
    resp = client.post("/api/v1/chat/attachments", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "rock.jpg"
    saved = Path(body["path"])
    assert saved.is_file()
    saved.unlink()


def test_upload_rejects_non_image(client):
    files = {"image": ("notes.txt", b"hello", "text/plain")}
    resp = client.post("/api/v1/chat/attachments", files=files)
    assert resp.status_code == 415


def test_upload_rejects_empty_file(client):
    files = {"image": ("empty.jpg", b"", "image/jpeg")}
    resp = client.post("/api/v1/chat/attachments", files=files)
    assert resp.status_code == 400
