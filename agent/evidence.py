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
        height, width = clip_frames[0].shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = directory / f"{prefix}_{timestamp}.avi"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (width, height))
        for frame in clip_frames:
            writer.write(frame)
        writer.release()
        return str(output)
