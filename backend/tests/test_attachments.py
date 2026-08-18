import time
from pathlib import Path

from app.services.attachments import (
    ATTACHMENT_DIR,
    cleanup_old_attachments,
    is_allowed_attachment,
    save_attachment,
)


def test_save_attachment_writes_file_inside_attachment_dir():
    path = save_attachment(b"fake-bytes", "rock.jpg")
    assert path.exists()
    assert path.read_bytes() == b"fake-bytes"
    assert ATTACHMENT_DIR in path.parents
    path.unlink()


def test_save_attachment_does_not_collide_on_same_name():
    a = save_attachment(b"one", "rock.jpg")
    b = save_attachment(b"two", "rock.jpg")
    assert a != b
    assert a.read_bytes() == b"one"
    assert b.read_bytes() == b"two"
    a.unlink()
    b.unlink()


def test_save_attachment_ignores_directory_components_in_filename():
    path = save_attachment(b"x", "../../etc/passwd")
    assert ATTACHMENT_DIR in path.parents
    path.unlink()


def test_is_allowed_attachment_accepts_saved_file():
    path = save_attachment(b"x", "rock.jpg")
    assert is_allowed_attachment(str(path)) is True
    path.unlink()


def test_is_allowed_attachment_rejects_paths_outside_the_dir():
    assert is_allowed_attachment("/etc/passwd") is False
    assert is_allowed_attachment(str(ATTACHMENT_DIR / ".." / "secret.txt")) is False


def test_is_allowed_attachment_rejects_existing_file_outside_the_dir(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    try:
        assert is_allowed_attachment(str(outside)) is False
    finally:
        outside.unlink()


def test_is_allowed_attachment_rejects_missing_file():
    assert is_allowed_attachment(str(ATTACHMENT_DIR / "does-not-exist.jpg")) is False


def test_cleanup_removes_only_old_files():
    old = save_attachment(b"old", "old.jpg")
    fresh = save_attachment(b"fresh", "fresh.jpg")
    past = time.time() - 7200
    import os

    os.utime(old, (past, past))

    removed = cleanup_old_attachments(max_age_seconds=3600)

    assert removed >= 1
    assert not old.exists()
    assert fresh.exists()
    fresh.unlink()
