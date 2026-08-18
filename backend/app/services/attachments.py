"""Temp storage for images attached in chat.

The agent's `classify_image` tool opens a filesystem path, so browser uploads
are written here and the path is handed to the agent. Only paths inside
ATTACHMENT_DIR are ever accepted back, so a crafted request cannot make the
agent open arbitrary files.
"""

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

logger = logging.getLogger("cv-agent-backend")

# Default to the platform temp dir so local runs on Windows do not create C:\tmp.
# docker-compose overrides ATTACHMENT_DIR for the container.
ATTACHMENT_DIR = Path(
    os.getenv("ATTACHMENT_DIR") or Path(tempfile.gettempdir()) / "cv-agent-attachments"
)


def _ensure_dir() -> None:
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)


def save_attachment(data: bytes, filename: str) -> Path:
    """Write bytes to a unique file inside ATTACHMENT_DIR and return its path."""
    _ensure_dir()
    # basename strips any directory components a client tried to smuggle in.
    safe_name = os.path.basename(filename) or "upload"
    path = ATTACHMENT_DIR / f"{uuid.uuid4().hex}-{safe_name}"
    path.write_bytes(data)
    return path


def is_allowed_attachment(path: str) -> bool:
    """True only for an existing regular file directly inside ATTACHMENT_DIR."""
    try:
        resolved = Path(path).resolve()
        base = ATTACHMENT_DIR.resolve()
    except Exception:
        return False

    if not resolved.is_file():
        return False
    return resolved.parent == base


def cleanup_old_attachments(max_age_seconds: int = 3600) -> int:
    """Delete attachments older than max_age_seconds. Returns how many went."""
    if not ATTACHMENT_DIR.exists():
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0
    for entry in ATTACHMENT_DIR.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except Exception as e:
            logger.warning("Could not remove attachment %s: %s", entry, e)
    return removed
