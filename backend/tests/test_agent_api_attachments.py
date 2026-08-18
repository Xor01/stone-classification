from unittest.mock import patch

from app.services.attachments import save_attachment


def test_attachment_path_is_described_to_the_agent(client):
    saved = save_attachment(b"bytes", "rock.jpg")

    with patch("app.api.agent_api.run_agent", return_value="Basalt") as mock_run:
        resp = client.post(
            "/api/v1/agent/chat",
            json={
                "message": "what is this?",
                "session_id": "s-1",
                "attachment_path": str(saved),
            },
        )

    assert resp.status_code == 200
    query = mock_run.call_args.kwargs["query"]
    assert "what is this?" in query
    assert str(saved) in query
    saved.unlink()


def test_attachment_path_outside_the_dir_is_rejected(client):
    with patch("app.api.agent_api.run_agent", return_value="ok") as mock_run:
        resp = client.post(
            "/api/v1/agent/chat",
            json={"message": "read this", "attachment_path": "/etc/passwd"},
        )

    assert resp.status_code == 400
    mock_run.assert_not_called()


def test_chat_without_attachment_is_unchanged(client):
    with patch("app.api.agent_api.run_agent", return_value="hi") as mock_run:
        resp = client.post("/api/v1/agent/chat", json={"message": "hello"})

    assert resp.status_code == 200
    assert mock_run.call_args.kwargs["query"] == "hello"
