import os
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



def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Returns a Langfuse CallbackHandler if credentials (LANGFUSE_PUBLIC_KEY & LANGFUSE_SECRET_KEY)
    are present in the environment; otherwise returns None.
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
    langfuse_handler = get_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
        tags=tags or ["cv-agent-production"],
        metadata=metadata,
    )
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

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
