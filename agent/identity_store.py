from __future__ import annotations

import json
from pathlib import Path

from agent.paths import writable_root


def _path(session_id: str) -> Path:
    folder = writable_root() / "identity"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{session_id}.json"


def load_embedding(session_id: str) -> list[float] | None:
    path = _path(session_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        vector = data.get("vector")
        if isinstance(vector, list) and vector:
            return [float(item) for item in vector]
    except Exception:
        return None
    return None


def save_embedding(session_id: str, vector: list[float]) -> None:
    _path(session_id).write_text(json.dumps({"vector": vector}), encoding="utf-8")
