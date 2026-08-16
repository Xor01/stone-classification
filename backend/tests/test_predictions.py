import io
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from PIL import Image


def _fake_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_valid_image_creates_record(client):
    files = {"image": ("cat.jpg", _fake_image_bytes(), "image/jpeg")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_class" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["inference_ms"] >= 0

    history = client.get("/api/v1/predictions").json()
    assert history["total"] == 1
    assert history["items"][0]["predicted_class"] == body["predicted_class"]


def test_predict_traces_the_classification(client):
    """Frontend uploads bypass the agent, so /predict must trace on its own."""
    files = {"image": ("rock.jpg", _fake_image_bytes(), "image/jpeg")}
    recorded = {}

    @contextmanager
    def fake_trace(*, image_name, content_type, size_bytes):
        recorded["input"] = (image_name, content_type, size_bytes)
        handle = MagicMock()
        yield handle
        recorded["handle"] = handle

    with patch("app.api.predictions.trace_classification", fake_trace):
        resp = client.post("/api/v1/predict", files=files)

    assert resp.status_code == 200
    image_name, content_type, size_bytes = recorded["input"]
    assert image_name == "rock.jpg"
    assert content_type == "image/jpeg"
    assert size_bytes > 0

    # The result must be recorded while the span is still open, otherwise the
    # span reports 0ms latency instead of the real inference time.
    recorded["handle"].record.assert_called_once()
    prediction = recorded["handle"].record.call_args.args[0]
    assert prediction.predicted_class == resp.json()["predicted_class"]


def test_tracing_failure_never_breaks_a_request(client):
    """A Langfuse outage must not turn a working prediction into a 500."""
    files = {"image": ("rock.jpg", _fake_image_bytes(), "image/jpeg")}

    with patch(
        "app.observability.get_langfuse_client", side_effect=RuntimeError("langfuse down")
    ):
        resp = client.post("/api/v1/predict", files=files)

    assert resp.status_code == 200


def test_predict_rejects_invalid_file_type(client):
    files = {"image": ("doc.txt", b"not an image", "text/plain")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 415


def test_prediction_by_id_not_found(client):
    resp = client.get("/api/v1/predictions/999")
    assert resp.status_code == 404


def test_stats_after_predictions(client):
    files = {"image": ("cat.jpg", _fake_image_bytes(), "image/jpeg")}
    client.post("/api/v1/predict", files=files)
    client.post("/api/v1/predict", files=files)

    resp = client.get("/api/v1/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_predictions"] == 2
    assert sum(body["class_distribution"].values()) == 2
