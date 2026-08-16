"""
Diagnostic script to test Langfuse authentication, OpenWebUI tool tracing, and AI Agent tracing locally.
Run with:
    uv run python scripts/test_langfuse.py
or:
    .venv\\Scripts\\python.exe scripts/test_langfuse.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

load_dotenv()

from langfuse import Langfuse
from agent.agent import get_langfuse_client, get_langfuse_handler, run_agent
from openwebui.openwebui_tools import Tools


def main():
    print("=" * 60)
    print("1. Testing Langfuse Authentication...")
    print("=" * 60)
    lf = get_langfuse_client()
    if not lf:
        print("[FAIL] Langfuse client failed to initialize. Check LANGFUSE_PUBLIC_KEY & LANGFUSE_SECRET_KEY in .env.")
        return

    is_valid = lf.auth_check()
    if is_valid:
        print("[OK] Langfuse Authentication SUCCESSFUL! Keys are valid.")
    else:
        print("[FAIL] Langfuse auth_check failed. Verify credentials.")
        return

    print("\n" + "=" * 60)
    print("2. Testing OpenWebUI Tool Tracing...")
    print("=" * 60)
    tools = Tools()
    tools._log_langfuse(
        "local_verification_tool",
        {"input": "Testing OpenWebUI tracing locally"},
        {"status": "Trace logged successfully"}
    )
    print("[OK] Sent tool observation to Langfuse Cloud!")

    print("\n" + "=" * 60)
    print("3. Testing AI Agent with Langfuse Tracing...")
    print("=" * 60)
    response = run_agent(
        "Who are you and what are your capabilities?",
        session_id="local-diagnostic-session",
        user_id="developer-test",
        tags=["local-verification"]
    )
    print("[OK] Agent response received:")
    print(f"    {response.strip()}")
    print("\n[SUCCESS] All Langfuse checks passed! Check your dashboard at https://cloud.langfuse.com")


if __name__ == "__main__":
    main()
