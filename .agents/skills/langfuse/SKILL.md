---
name: langfuse
description: Interact with Langfuse and access its documentation. Use when needing to (1) query or modify Langfuse data programmatically via the CLI, (2) look up Langfuse documentation, concepts, integration guides, or SDK usage, or (3) instrument LLM applications with Langfuse observability and tracing.
---

# Langfuse Skill

This skill provides best practices for instrumenting applications with Langfuse tracing, evaluation, and prompt management.

## Core Principles

1. **Framework Integration First**: For LangChain, use `from langfuse.langchain import CallbackHandler` or `from langfuse.callback import CallbackHandler`.
2. **Environment Variables**:
   - `LANGFUSE_PUBLIC_KEY="pk-lf-..."`
   - `LANGFUSE_SECRET_KEY="sk-lf-..."`
   - `LANGFUSE_HOST="https://cloud.langfuse.com"` (or self-hosted / regional endpoint)
3. **Trace Metadata**: Include useful metadata such as session_id, user_id, and tags.
4. **Graceful Fallback**: If Langfuse credentials are not provided, tracing should be optional and not crash the application.
