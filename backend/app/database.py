"""Database engine/session management (SQLModel + Postgres)."""

from collections.abc import Generator

from sqlmodel import SQLModel, Session, create_engine

from app.config import get_settings

settings = get_settings()

# pool_pre_ping avoids stale-connection errors after container restarts
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, echo=False)


def init_db() -> None:
    """Create tables if they don't exist. Called on app startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session per request."""
    with Session(engine) as session:
        yield session
