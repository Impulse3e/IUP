from __future__ import annotations

import sys
import time


_ALLOWED_TITLE = (
    "iup proctoring",
    "iup —",
    "iup -",
    "chrome",
    "edge",
    "firefox",
    "opera",
    "yandex",
    "brave",
    "exam",
    "moodle",
    "тест",
    "участник",
    "lockdown",
    "chromium",
    "msedge",
)

_ALLOWED_PROCESS = (
    "chrome",
    "msedge",
    "firefox",
    "opera",
    "brave",
    "yandex",
    "iexplore",
    "python",
    "pythonw",
    "iup",
)

_cached_at = 0.0
_cached_lost = False


def camera_window_minimized() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    hwnd = ctypes.windll.user32.FindWindowW(None, "IUP Proctoring")
    if not hwnd:
        return False
    return bool(ctypes.windll.user32.IsIconic(hwnd))


def window_focus_lost() -> bool:
    """True when neither the camera window nor a typical exam browser is in front."""
    global _cached_at, _cached_lost
    now = time.monotonic()
    if now - _cached_at < 0.3:
        return _cached_lost
    lost = _detect_focus_lost()
    _cached_at = now
    _cached_lost = lost
    return lost


def _detect_focus_lost() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return _cached_lost
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    title = (buf.value or "").strip().lower()
    if any(part in title for part in _ALLOWED_TITLE):
        return False
    try:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        import psutil

        name = (psutil.Process(pid.value).name() or "").lower()
        if any(part in name for part in _ALLOWED_PROCESS):
            return False
    except Exception:
        if not title:
            return _cached_lost
    if camera_window_minimized() and not title:
        return _cached_lost
    return bool(title)


def foreground_window() -> dict[str, str]:
    """Active window title and process name. Empty on non-Windows."""
    if sys.platform != "win32":
        return {"title": "", "process": ""}
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return {"title": "", "process": ""}
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    title = (buf.value or "").strip()
    process = ""
    try:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        import psutil

        process = (psutil.Process(pid.value).name() or "").strip()
    except Exception:
        process = ""
    return {"title": title, "process": process}


def match_watch_title(title: str, needles: list[str]) -> str | None:
    hay = (title or "").strip().lower()
    if not hay or "iup proctoring" in hay:
        return None
    for needle in needles:
        item = (needle or "").strip().lower()
        if len(item) >= 2 and item in hay:
            return item
    return None
