from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from server.app.config import settings


def sqlite_path() -> Path:
    url = settings.database_url
    if not url.startswith("sqlite:///"):
        raise RuntimeError("Автобэкап поддерживается только для SQLite")
    raw = url.removeprefix("sqlite:///")
    if not raw or raw == ":memory:":
        raise RuntimeError("Нельзя копировать in-memory базу")
    return Path(raw)


def backup_database(keep: int = 14) -> Path:
    src = sqlite_path()
    if not src.is_file():
        raise FileNotFoundError(f"База не найдена: {src}")
    dest_dir = src.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"iup-{datetime.utcnow():%Y%m%d-%H%M%S}.db"
    source = sqlite3.connect(str(src))
    try:
        target = sqlite3.connect(str(dest))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    old = sorted(dest_dir.glob("iup-*.db"), key=lambda path: path.stat().st_mtime, reverse=True)
    for extra in old[keep:]:
        extra.unlink(missing_ok=True)
    return dest


if __name__ == "__main__":
    path = backup_database()
    print(path)
