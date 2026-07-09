import platform
import subprocess

import psutil

from shared.constants import FORBIDDEN_PROCESSES


def list_forbidden_processes() -> list[str]:
    found = []
    for proc in psutil.process_iter(["name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            exe = (proc.info.get("exe") or "").lower()
            for forbidden in FORBIDDEN_PROCESSES:
                if forbidden in name or forbidden in exe:
                    found.append(proc.info["name"] or forbidden)
                    break
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
