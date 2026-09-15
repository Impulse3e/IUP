from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class EvidenceBuffer:
    def __init__(self, seconds: int = 10, fps: int = 8) -> None:
        self.maxlen = seconds * fps
        self.frames: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._skip = 0

    def push(self, frame: np.ndarray) -> None:
        self._skip = (self._skip + 1) % 2
        if self._skip:
            return
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            frame = cv2.resize(frame, (640, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        self.frames.append(frame.copy())

    def save_clip(self, directory: Path, prefix: str, frames: list[np.ndarray] | None = None) -> str | None:
        clip_frames = list(self.frames if frames is None else frames)
        if not clip_frames:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = directory / f"{prefix}_{timestamp}.jpg"
        step = max(1, len(clip_frames) // 6)
        sampled = clip_frames[::step][:6]
        jpeg = encode_jpeg_strip(sampled)
        if not jpeg:
            return None
        output.write_bytes(jpeg)
        return str(output)


def encode_jpeg_strip(frames: list[np.ndarray], quality: int = 80) -> bytes | None:
    if not frames:
        return None
    thumbs: list[np.ndarray] = []
    for frame in frames[:6]:
        if frame is None or getattr(frame, "size", 0) == 0:
            continue
        height, width = frame.shape[:2]
        scale = 320 / max(width, 1)
        thumbs.append(cv2.resize(frame, (320, max(1, int(height * scale))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return None
    height = max(item.shape[0] for item in thumbs)

    def pad_height(image: np.ndarray, target: int) -> np.ndarray:
        if image.shape[0] >= target:
            return image
        return cv2.copyMakeBorder(image, 0, target - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)

    def pad_width(image: np.ndarray, target: int) -> np.ndarray:
        if image.shape[1] >= target:
            return image
        return cv2.copyMakeBorder(image, 0, 0, 0, target - image.shape[1], cv2.BORDER_CONSTANT)

    thumbs = [pad_height(item, height) for item in thumbs]
    if len(thumbs) <= 3:
        mosaic = np.hstack(thumbs)
    else:
        mid = (len(thumbs) + 1) // 2
        row1 = np.hstack(thumbs[:mid])
        row2 = np.hstack(thumbs[mid:])
        width = max(row1.shape[1], row2.shape[1])
        mosaic = np.vstack([pad_width(row1, width), pad_width(row2, width)])
    ok, buffer = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def encode_jpeg(frame: np.ndarray | None, quality: int = 55, max_width: int = 1280) -> bytes | None:
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    image = frame
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None
