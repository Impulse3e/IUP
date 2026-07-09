import numpy as np


def screen_size() -> tuple[int, int]:
    try:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            return int(monitor["width"]), int(monitor["height"])
    except Exception:
        import pyautogui

        size = pyautogui.size()
        return int(size.width), int(size.height)


def screenshot_bgr() -> np.ndarray:
    import cv2

    errors: list[str] = []

    try:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            image = np.array(sct.grab(monitor))
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    except Exception as error:
        errors.append(f"mss: {error}")

    try:
        import pyautogui

        image = pyautogui.screenshot()
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as error:
        errors.append(f"pyautogui: {error}")

    raise RuntimeError("Не удалось захватить экран. " + "; ".join(errors))


def screen_available() -> bool:
    try:
        screenshot_bgr()
        return True
    except Exception:
        return False
