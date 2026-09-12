import secrets
import string
from pathlib import Path

ALLOWED_CHUNK_SOURCES = {"webcam", "screen", "preview"}
ALLOWED_EVIDENCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".avi", ".mp4", ".webm", ".bin"}
STAFF_ROLES = {"proctor", "teacher", "admin"}


def generate_temp_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_EVIDENCE_SUFFIXES:
        return suffix
    return ".bin"


def assert_safe_relative(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts or not normalized:
        raise ValueError("Invalid storage path")
    return normalized
