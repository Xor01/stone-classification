import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

from app import database
from app.main import app


@pytest.fixture(autouse=True)
def _use_test_db(tmp_path, monkeypatch):
    """Point the app at a throwaway SQLite DB for each test."""
    db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(database, "engine", test_engine)
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
