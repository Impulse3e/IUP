from __future__ import annotations

import secrets
from pathlib import Path

from server.app.config import settings

DEFAULT_SECRET = "change-me-in-production"
SECRET_FILE = Path("data/secret.key")


def ensure_secret_key() -> None:
    """Use .env if set, otherwise reuse or create data/secret.key."""
    current = (settings.secret_key or "").strip()
    if current and current != DEFAULT_SECRET:
        return
    if SECRET_FILE.is_file():
        stored = SECRET_FILE.read_text(encoding="utf-8").strip()
        if stored:
            settings.secret_key = stored
            return
    generated = secrets.token_urlsafe(48)
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRET_FILE.write_text(generated, encoding="utf-8")
    settings.secret_key = generated
    print("IUP: created data/secret.key for JWT signing.")
