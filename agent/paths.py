"""Cross-platform paths for IUP student launcher and agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"
    return base / "iup" / "launcher.json"


def venv_python(root: Path | None = None) -> Path | None:
    root = root or project_root()
    if sys.platform == "win32":
        candidate = root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else None


def venv_pythonw(root: Path | None = None) -> Path | None:
    if sys.platform != "win32":
        return venv_python(root)
    root = root or project_root()
    candidate = root / ".venv" / "Scripts" / "pythonw.exe"
    return candidate if candidate.exists() else venv_python(root)


def agent_executable(root: Path | None = None) -> Path | None:
    root = root or project_root()
    for name in ("IUP-Agent.exe", "iup-agent.exe"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def agent_python(root: Path | None = None) -> Path:
    return venv_python(root) or Path(sys.executable)


def install_path() -> Path:
    env = os.environ.get("IUP_INSTALL_PATH")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "IUP"
    return Path.home() / "IUP"


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def writable_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "iup"
    else:
        base = Path.home() / ".local" / "share" / "iup"
    base.mkdir(parents=True, exist_ok=True)
    return base


def model_dir() -> Path:
    bundled = resource_root() / "models" / "face_landmarker.task"
    if bundled.exists():
        return bundled.parent
    return project_root() / "models"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)
