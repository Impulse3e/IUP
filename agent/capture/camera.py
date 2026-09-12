from __future__ import annotations

import sys
import threading
import time

import cv2


WINDOW_NAME = "IUP Proctoring"
_window_ready = False


def open_camera(index: int = 0):
    backends = []
    if sys.platform == "win32":
        backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_MSMF)
    backends.append(cv2.CAP_ANY)

    last = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap
        cap.release()
        last = cap
        time.sleep(0.2)

    if last is not None:
        return last
    return cv2.VideoCapture(index)


def show_camera_window(frame) -> None:
    global _window_ready
    if not _window_ready:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        _window_ready = True
    cv2.imshow(WINDOW_NAME, frame)


def call_with_camera_pump(cap, fn, message: str):
    """Keep the OpenCV window alive while a blocking call runs in another thread."""
    done = threading.Event()
    box: dict = {}

    def work() -> None:
        try:
            box["result"] = fn()
        except Exception as error:
            box["error"] = error
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    while not done.wait(0.03):
        success, frame = cap.read()
        if success and frame is not None:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (30, 30, 30), -1)
            cv2.putText(frame, message, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            show_camera_window(frame)
        cv2.waitKey(30)
    if "error" in box:
        raise box["error"]
    return box.get("result")
