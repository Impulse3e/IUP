from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from server.app.config import settings


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_parent() -> None:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        return
    db_path = url.removeprefix("sqlite:///")
    if not db_path or db_path == ":memory:":
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from server.app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
