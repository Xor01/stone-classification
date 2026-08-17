import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Ensure root is in sys.path for importing agent module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.agent import agent, run_agent
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    classify_image,
    get_model_info,
    get_prediction_by_id,
    get_prediction_history,
    get_prediction_statistics,
    tools,
)


def test_agent_tools_list():
    """Verify all 5 required tools are registered."""
    tool_names = [t.name for t in tools]
    assert "classify_image" in tool_names
    assert "get_prediction_history" in tool_names
    assert "get_prediction_by_id" in tool_names
    assert "get_prediction_statistics" in tool_names
    assert "get_model_info" in tool_names
    assert len(tools) == 5


def test_system_prompt_rules():
    """Verify system prompt contains essential agent rules."""
    assert "Never invent prediction results" in SYSTEM_PROMPT
    assert "tools" in SYSTEM_PROMPT.lower()


@patch("httpx.get")
def test_get_model_info_tool(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "model_name": "cv-agent-classifier",
        "version": "1.0.0",
        "classes": {"0": "art_1", "1": "art_2"},
        "deployment_status": "loaded",
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_model_info.invoke({})
    assert result["model_name"] == "cv-agent-classifier"
    assert result["version"] == "1.0.0"
    mock_get.assert_called_once()


@patch("httpx.get")
def test_get_prediction_history_tool(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "total": 1,
        "items": [
            {
                "id": 1,
                "image_name": "test.jpg",
                "predicted_class": "art_1",
                "confidence": 0.95,
                "created_at": "2026-08-15T12:00:00",
            }
        ],
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_prediction_history.invoke({"limit": 5, "offset": 0})
    assert result["total"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["predicted_class"] == "art_1"


@patch("httpx.get")
def test_get_prediction_by_id_tool(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "id": 42,
        "image_name": "painting.jpg",
        "predicted_class": "renaissance",
        "confidence": 0.98,
        "created_at": "2026-08-15T12:00:00",
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_prediction_by_id.invoke({"prediction_id": 42})
    assert result["id"] == 42
    assert result["predicted_class"] == "renaissance"


@patch("httpx.get")
def test_get_prediction_statistics_tool(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "total_predictions": 50,
        "class_distribution": {"cubism": 20, "impressionism": 30},
        "average_confidence": 0.92,
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    result = get_prediction_statistics.invoke({})
    assert result["total_predictions"] == 50
    assert result["class_distribution"]["cubism"] == 20


@patch("httpx.post")
def test_classify_image_tool(mock_post, tmp_path):
    # Create a dummy test image
    test_img = tmp_path / "sample.jpg"
    test_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "predicted_class": "baroque",
        "confidence": 0.94,
        "top_predictions": [{"class_name": "baroque", "probability": 0.94}],
        "inference_ms": 25.0,
        "model_version": "1.0.0",
    }
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    result = classify_image.invoke({"image_path": str(test_img)})
    assert result["predicted_class"] == "baroque"
    assert result["confidence"] == 0.94
    mock_post.assert_called_once()


@patch("agent.agent.agent.invoke")
def test_run_agent_helper(mock_invoke):
    mock_msg = MagicMock()
    mock_msg.content = "There are 50 total predictions in the database."
    mock_invoke.return_value = {"messages": [mock_msg]}

    reply = run_agent("How many predictions were made?")
    assert "50 total predictions" in reply


def test_get_langfuse_handler_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    import agent.agent

    agent.agent._langfuse_client = None
    handler = agent.agent.get_langfuse_handler()
    assert handler is None


def test_get_langfuse_handler_initializes_when_keys_present(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    import agent.agent

    agent.agent._langfuse_client = None

    with patch("langfuse.Langfuse") as mock_lf, patch(
        "langfuse.langchain.CallbackHandler"
    ) as mock_cb:
        mock_lf.return_value = MagicMock()
        mock_cb.return_value = MagicMock()
        handler = agent.agent.get_langfuse_handler()
        assert handler is not None
        mock_lf.assert_called_once_with(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",
            host="https://cloud.langfuse.com",
        )
        mock_cb.assert_called_once()


def test_run_agent_propagates_trace_attributes(monkeypatch):
    """Session/user/tags must reach Langfuse, else traces land as anonymous runs."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    import agent.agent

    agent.agent._langfuse_client = None

    mock_msg = MagicMock()
    mock_msg.content = "done"

    with patch("langfuse.Langfuse"), patch("langfuse.langchain.CallbackHandler"), patch(
        "langfuse.propagate_attributes"
    ) as mock_propagate, patch.object(
        agent.agent.agent, "invoke", return_value={"messages": [mock_msg]}
    ):
        agent.agent.run_agent(
            "hello", session_id="s-1", user_id="u-1", tags=["custom-tag"]
        )

    mock_propagate.assert_called_once_with(
        trace_name="cv-agent-chat",
        session_id="s-1",
        user_id="u-1",
        tags=["custom-tag"],
        metadata=None,
        environment="development",
    )


