import os
from contextlib import nullcontext

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from agent.tools import tools
from agent.prompts import SYSTEM_PROMPT

load_dotenv()


def get_llm():
    model_name = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "mock-key"
    base_url = os.getenv("LLM_BASE_URL")

    kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


model = get_llm()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

_langfuse_client = None


def get_langfuse_client():
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip("\"' ")
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip("\"' ")
    host = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    ).strip("\"' ")

    if not public_key or not secret_key:
        return None

    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        return _langfuse_client
    except Exception as e:
        import logging

        logging.getLogger("cv-agent").warning(
            "Failed to initialize Langfuse client: %s", str(e)
        )
        return None



def get_langfuse_handler():
    """
    Returns a Langfuse CallbackHandler if credentials (LANGFUSE_PUBLIC_KEY & LANGFUSE_SECRET_KEY)
    are present in the environment; otherwise returns None.

    Trace-level attributes (name, session, user, tags) are NOT set here - in the
    v4 SDK the handler takes no such arguments. They are applied by `run_agent`
    via `propagate_attributes`.
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as e:
        import logging

        logging.getLogger("cv-agent").warning(
            "Failed to initialize Langfuse CallbackHandler: %s", str(e)
        )
        return None


def run_agent(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Convenience function to run a natural language query through the agent with Langfuse tracing."""
    config = {}
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    # Without this the trace lands as an anonymous "LangGraph" run with no
    # session/user/tags, which is indistinguishable from tracing being broken.
    trace_context = nullcontext()
    if langfuse_handler:
        from langfuse import propagate_attributes

        # `environment` (not a hardcoded "production" tag) is what keeps dev and
        # prod traffic separate - the old default tagged local runs as production.
        trace_context = propagate_attributes(
            trace_name="cv-agent-chat",
            session_id=session_id,
            user_id=user_id,
            tags=tags or ["cv-agent"],
            metadata=metadata,
            environment=os.getenv("ENV", "development"),
        )

    with trace_context:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config if config else None,
        )

    client = get_langfuse_client()
    if client and hasattr(client, "flush"):
        try:
            client.flush()
        except Exception:
            pass

    messages = response.get("messages", [])
    if messages:
        last_message = messages[-1]
        return getattr(last_message, "content", str(last_message))
    return "No response generated."
