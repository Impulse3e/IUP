import platform
import re
import subprocess
from pathlib import Path

import psutil

from shared.constants import FORBIDDEN_PROCESSES


def _tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        name = Path(value).name.lower()
        stem = Path(name).stem.lower()
        tokens.add(stem)
        tokens.update(part for part in re.split(r"[^a-z0-9]+", stem) if part)
    return tokens


def list_forbidden_processes() -> list[str]:
    found = []
    for proc in psutil.process_iter(["name"]):
        try:
            name = proc.info.get("name") or ""
            tokens = _tokens(name)
            if tokens & FORBIDDEN_PROCESSES:
                found.append(name or next(iter(tokens & FORBIDDEN_PROCESSES)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(set(found))


def monitor_count() -> int:
    if platform.system() == "Windows":
        try:
            import ctypes

            return int(ctypes.windll.user32.GetSystemMetrics(80))
        except Exception:
            return 1
    try:
        output = subprocess.check_output(["xrandr", "--query"], text=True, stderr=subprocess.DEVNULL)
        return sum(1 for line in output.splitlines() if " connected " in line)
    except Exception:
        return 1
