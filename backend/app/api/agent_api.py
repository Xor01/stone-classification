import logging
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from agent.agent import run_agent

from app.services.attachments import is_allowed_attachment


logger = logging.getLogger("cv-agent-backend")

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User message or prompt for the AI agent")
    session_id: str | None = Field(None, description="Optional session ID for tracing and thread continuity")
    user_id: str | None = Field(None, description="Optional user identifier for tracing")
    attachment_path: str | None = Field(
        None, description="Server-side path of an image uploaded via /api/v1/chat/attachments"
    )


class AgentChatResponse(BaseModel):
    response: str = Field(..., description="Agent response after invoking necessary tools")


@router.post("/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    Send a natural language message to the AI Assistant.
    The assistant can call real backend tools (model info, stats, history, classification) to fulfill the request.
    Traces are automatically logged to Langfuse if configured.
    """
    query = request.message
    if request.attachment_path:
        if not is_allowed_attachment(request.attachment_path):
            raise HTTPException(
                status_code=400, detail="Unknown or invalid attachment"
            )
        # Give the LLM a concrete path so it can call classify_image itself.
        query = f"{query}\n\nThe user attached an image at {request.attachment_path}."

    try:
        reply = run_agent(
            query=query,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return AgentChatResponse(response=reply)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent chat execution failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

