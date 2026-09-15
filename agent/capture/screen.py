import sys

import numpy as np

_sct = None
_dpi_ready = False


def _mss():
    global _sct
    import mss

    if _sct is None:
        _sct = mss.mss()
    return _sct


def _ensure_dpi_aware() -> None:
    global _dpi_ready
    if _dpi_ready or sys.platform != "win32":
        return
    try:
        ctypes = __import__("ctypes")
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes = __import__("ctypes")
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    _dpi_ready = True


def _is_mostly_black(image: np.ndarray, limit: float = 10.0) -> bool:
    if image is None or getattr(image, "size", 0) == 0:
        return True
    height, width = image.shape[:2]
    step_y = max(1, height // 64)
    step_x = max(1, width // 64)
    sample = image[::step_y, ::step_x]
    return float(sample.mean()) < limit


def screen_size() -> tuple[int, int]:
    try:
        sct = _mss()
        monitors = sct.monitors
        monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        return int(monitor["width"]), int(monitor["height"])
    except Exception:
        import pyautogui

        size = pyautogui.size()
        return int(size.width), int(size.height)


def _grab_mss() -> np.ndarray:
    import cv2

    sct = _mss()
    monitors = sct.monitors
    monitor = monitors[1] if len(monitors) > 1 else monitors[0]
    image = np.array(sct.grab(monitor))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def _grab_pyautogui() -> np.ndarray:
    import cv2
    import pyautogui

    image = pyautogui.screenshot()
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _grab_foreground_window() -> np.ndarray:
    """Capture the active window, including GPU-composited Chrome/Edge tabs."""
    if sys.platform != "win32":
        raise RuntimeError("foreground capture is Windows-only")
    import ctypes
    from ctypes import wintypes

    import cv2

    _ensure_dpi_aware()
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        raise RuntimeError("no foreground window")

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetWindowRect failed")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 16 or height < 16 or width > 7680 or height > 4320:
        raise RuntimeError(f"bad window size {width}x{height}")

    PW_RENDERFULLCONTENT = 2
    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        raise RuntimeError("GetWindowDC failed")
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old = gdi32.SelectObject(mem_dc, bitmap)
    try:
        painted = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not painted:
            painted = user32.PrintWindow(hwnd, mem_dc, 0)
        if not painted:
            raise RuntimeError("PrintWindow failed")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0
        image_size = width * height * 4
        header.biSizeImage = image_size
        buf = (ctypes.c_ubyte * image_size)()
        gdi32.SelectObject(mem_dc, old)
        old = None
        bits = gdi32.GetDIBits(hwnd_dc, bitmap, 0, height, buf, ctypes.byref(header), 0)
        if bits == 0:
            raise RuntimeError("GetDIBits failed")
        image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4)).copy()
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    finally:
        if old is not None:
            gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


def screenshot_bgr(*, prefer_foreground: bool = True) -> np.ndarray:
    errors: list[str] = []
    frames: list[np.ndarray] = []

    if prefer_foreground:
        try:
            frame = _grab_foreground_window()
            if not _is_mostly_black(frame):
                return frame
            frames.append(frame)
        except Exception as error:
            errors.append(f"window: {error}")

    try:
        frame = _grab_mss()
        if not _is_mostly_black(frame):
            return frame
        frames.append(frame)
    except Exception as error:
        errors.append(f"mss: {error}")

    try:
        frame = _grab_pyautogui()
        if not _is_mostly_black(frame):
            return frame
        frames.append(frame)
    except Exception as error:
        errors.append(f"pyautogui: {error}")

    if frames:
        return max(frames, key=lambda item: float(item.mean()))
    raise RuntimeError("Не удалось захватить экран. " + "; ".join(errors))


def screen_available() -> bool:
    try:
        screenshot_bgr(prefer_foreground=False)
        return True
    except Exception:
        return False
