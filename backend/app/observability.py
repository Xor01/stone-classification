"""Langfuse tracing for the API layer.

The agent path (`agent/agent.py`) traces itself via the LangChain callback
handler. Direct `/api/v1/predict` uploads never go through the agent, so they
are traced here instead - without this, image classifications from the frontend
are invisible in Langfuse.
"""

import logging
from contextlib import ExitStack, contextmanager
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger("cv-agent-backend")


@lru_cache
def get_langfuse_client():
    """Build the Langfuse client once, or return None if unconfigured."""
    settings = get_settings()

    public_key = (settings.LANGFUSE_PUBLIC_KEY or "").strip("\"' ")
    secret_key = (settings.LANGFUSE_SECRET_KEY or "").strip("\"' ")
    host = (settings.LANGFUSE_HOST or "https://cloud.langfuse.com").strip("\"' ")

    if not public_key or not secret_key:
        logger.info("Langfuse keys not configured; prediction tracing disabled.")
        return None

    try:
        from langfuse import Langfuse

        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as e:
        logger.warning("Failed to initialize Langfuse client: %s", e)
        return None


class _ClassificationTrace:
    """Handle for attaching a prediction result to an in-flight span."""

    def __init__(self, span):
        self._span = span

    def record(self, prediction) -> None:
        """Attach the model's output. No-op when tracing is disabled."""
        if self._span is None:
            return
        try:
            self._span.update(
                output={
                    "predicted_class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    "top_predictions": prediction.top_k_predictions or [],
                },
                metadata={
                    "model_version": prediction.model_version,
                    "request_id": prediction.request_id,
                    "prediction_id": prediction.id,
                    "inference_ms": prediction.inference_ms,
                },
            )
        except Exception as e:
            logger.warning("Langfuse span update failed: %s: %s", type(e).__name__, e)


@contextmanager
def trace_classification(*, image_name: str, content_type: str | None, size_bytes: int):
    """Wrap a classification so the span records the real inference latency.

    Metadata only - the uploaded image is deliberately not sent to Langfuse.
    A tracing failure must never fail a prediction request, so setup errors are
    logged and degrade to a no-op handle.
    """
    with ExitStack() as stack:
        span = None
        try:
            client = get_langfuse_client()
            if client is not None:
                from langfuse import propagate_attributes

                settings = get_settings()
                # `environment` keeps dev/test traffic out of production
                # analytics; without it everything lands in "default".
                stack.enter_context(
                    propagate_attributes(
                        trace_name="classify-image",
                        tags=["cv-backend", "predict"],
                        environment=settings.ENV,
                    )
                )
                span = stack.enter_context(
                    client.start_as_current_observation(
                        name="classify-image",
                        as_type="span",
                        input={
                            "image_name": image_name,
                            "content_type": content_type,
                            "size_kb": round(size_bytes / 1024, 1),
                        },
                    )
                )
        except Exception as e:
            logger.warning("Langfuse tracing setup failed: %s: %s", type(e).__name__, e)
            span = None

        # Body exceptions propagate: the ExitStack marks the span as failed and
        # the endpoint's own error handling still runs.
        yield _ClassificationTrace(span)


def flush_langfuse() -> None:
    """Flush pending traces on shutdown so short-lived containers don't drop them."""
    try:
        client = get_langfuse_client()
        if client is None:
            return
        client.flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
